use cudarc::{
    driver::{CudaContext, DeviceRepr, DriverError, LaunchConfig, PushKernelArg, ValidAsZeroBits},
    nvrtc::Ptx,
};

use rayon::prelude::*;

use honeycomb::prelude::{CMap2, CMapBuilder, CoordsFloat, Vertex2};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub const N_X: usize = 2048;
pub const N_Y: usize = 2048;
const LEN_CELL_X: f32 = 1.0;
const LEN_CELL_Y: f32 = 1.0;
pub const N_DARTS: usize = 1 + N_X * N_Y * 4;
const KERNEL: &str = include_str!(concat!(env!("OUT_DIR"), "/grid.ptx"));

const BLOCK_DIMS: (u32, u32, u32) = (4, 4, 4);
const GRID_DIMS: (u32, u32, u32) = (
    (N_X as u32).div_ceil(BLOCK_DIMS.0),
    (N_Y as u32).div_ceil(BLOCK_DIMS.1),
    1,
);

impl Default for CuVertex2 {
    fn default() -> Self {
        Self { data: [0.0; 2] }
    }
}
unsafe impl DeviceRepr for CuVertex2 {}
unsafe impl ValidAsZeroBits for CuVertex2 {}

impl<T: CoordsFloat> From<CuVertex2> for Vertex2<T> {
    fn from(value: CuVertex2) -> Self {
        let CuVertex2 { data: [x, y] } = value;
        Self(T::from(x).unwrap(), T::from(y).unwrap())
    }
}

pub fn build_gpu<T: CoordsFloat>() -> Result<CMap2<T>, DriverError> {
    let ctx = CudaContext::new(0)?;
    let mut betas = unsafe { ctx.alloc_pinned::<DartIdType>(3 * N_DARTS)? };
    let mut vertices = unsafe { ctx.alloc_pinned::<CuVertex2>(N_DARTS)? };

    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_src(KERNEL))?;
    let cfg = LaunchConfig {
        grid_dim: GRID_DIMS,
        block_dim: BLOCK_DIMS,
        shared_mem_bytes: 0,
    };
    {
        let st = stream.fork()?;
        let gen_beta = module.load_function("generate_2d_grid_betaf")?;
        let mut out_device = st.alloc_zeros::<DartIdType>(3 * N_DARTS)?;
        let mut launch_args = st.launch_builder(&gen_beta);
        launch_args.arg(&mut out_device);
        launch_args.arg(&N_X);
        launch_args.arg(&N_Y);
        launch_args.arg(&(3 * N_DARTS));
        unsafe { launch_args.launch(cfg.clone())? };

        st.memcpy_dtoh(&out_device, &mut betas)?;
    }
    {
        let st = stream.fork()?;
        let gen_vertices = module.load_function("generate_2d_grid_vertices")?;
        let mut out_device = st.alloc_zeros::<CuVertex2>(N_DARTS)?;
        let mut launch_args = st.launch_builder(&gen_vertices);
        launch_args.arg(&mut out_device);
        launch_args.arg(&LEN_CELL_X);
        launch_args.arg(&LEN_CELL_Y);
        launch_args.arg(&N_X);
        launch_args.arg(&N_Y);
        launch_args.arg(&N_DARTS);
        unsafe { launch_args.launch(cfg.clone())? };

        st.memcpy_dtoh(&out_device, &mut vertices)?;
    }
    let map: CMap2<T> = CMapBuilder::<2>::from_n_darts(N_DARTS - 1).build().unwrap();

    let betas = betas.as_slice()?;
    let bcs = betas.chunks(3).enumerate().collect::<Vec<_>>();
    bcs.into_par_iter().for_each(|(i, c)| {
        let d = i as DartIdType; // account for the null dart
        let &[b0, b1, b2] = c else { unreachable!() };
        map.set_betas(d, [b0, b1, b2]);
    });

    let vertices = vertices.as_slice()?;
    map.par_iter_vertices().for_each(|d| {
        map.force_write_vertex(d as VertexIdType, vertices[d as usize]);
    });
    Ok(map)
}
