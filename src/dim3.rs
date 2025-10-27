use cudarc::{
    driver::{CudaContext, DeviceRepr, DriverError, LaunchConfig, PushKernelArg, ValidAsZeroBits},
    nvrtc::Ptx,
};

use rayon::prelude::*;

use honeycomb::prelude::{CMap3, CMapBuilder, CoordsFloat, Vertex3};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub const N_X: usize = 100;
pub const N_Y: usize = 100;
pub const N_Z: usize = 100;
const LEN_CELL_X: f32 = 1.0;
const LEN_CELL_Y: f32 = 1.0;
const LEN_CELL_Z: f32 = 1.0;
pub const N_DARTS: usize = 1 + N_X * N_Y * N_Z * 24;
const KERNEL: &str = include_str!(concat!(env!("OUT_DIR"), "/grid.ptx"));

const BLOCK_DIMS: (u32, u32, u32) = (2, 2, 24);
const GRID_DIMS: (u32, u32, u32) = (
    (N_X as u32).div_ceil(BLOCK_DIMS.0),
    (N_Y as u32).div_ceil(BLOCK_DIMS.1),
    N_Z as u32,
);

impl Default for CuVertex3 {
    fn default() -> Self {
        Self { data: [0.0; 3] }
    }
}
unsafe impl DeviceRepr for CuVertex3 {}
unsafe impl ValidAsZeroBits for CuVertex3 {}

impl<T: CoordsFloat> From<CuVertex3> for Vertex3<T> {
    fn from(value: CuVertex3) -> Self {
        let CuVertex3 { data: [x, y, z] } = value;
        Self(
            T::from(x).unwrap(),
            T::from(y).unwrap(),
            T::from(z).unwrap(),
        )
    }
}

pub fn build_gpu<T: CoordsFloat>() -> Result<CMap3<T>, DriverError> {
    let ctx = CudaContext::new(0)?;
    let mut betas = unsafe { ctx.alloc_pinned::<DartIdType>(4 * N_DARTS)? };
    let mut vertices = unsafe { ctx.alloc_pinned::<CuVertex3>(N_DARTS)? };

    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_src(KERNEL))?;
    let cfg = LaunchConfig {
        grid_dim: GRID_DIMS,
        block_dim: BLOCK_DIMS,
        shared_mem_bytes: 0,
    };
    {
        let st = stream.fork()?;
        let gen_beta = module.load_function("generate_hex_grid_betaf")?;
        let mut out_device = st.alloc_zeros::<DartIdType>(4 * N_DARTS)?;
        let mut launch_args = st.launch_builder(&gen_beta);
        launch_args.arg(&mut out_device);
        launch_args.arg(&N_X);
        launch_args.arg(&N_Y);
        launch_args.arg(&N_Z);
        launch_args.arg(&(4 * N_DARTS));
        unsafe { launch_args.launch(cfg.clone())? };

        st.memcpy_dtoh(&out_device, &mut betas)?;
    }
    {
        let st = stream.fork()?;
        let gen_vertices = module.load_function("generate_hex_grid_vertices")?;
        let mut out_device = st.alloc_zeros::<CuVertex3>(N_DARTS)?;
        let mut launch_args = st.launch_builder(&gen_vertices);
        launch_args.arg(&mut out_device);
        launch_args.arg(&LEN_CELL_X);
        launch_args.arg(&LEN_CELL_Y);
        launch_args.arg(&LEN_CELL_Z);
        launch_args.arg(&N_X);
        launch_args.arg(&N_Y);
        launch_args.arg(&N_DARTS);
        unsafe { launch_args.launch(cfg.clone())? };

        st.memcpy_dtoh(&out_device, &mut vertices)?;
    }
    let map: CMap3<T> = CMapBuilder::<3>::from_n_darts(N_DARTS - 1).build().unwrap();

    let betas = betas.as_slice()?;
    // let betas: Vec<_> = betas
    //     .as_slice()?
    //     .chunks_exact(32)
    //     .flat_map(|c| &c[..24])
    //     .cloned()
    //     .collect();
    let bcs = betas.chunks_exact(4).enumerate().collect::<Vec<_>>();
    // println!("{}", bcs.len());
    // panic!();
    bcs.into_par_iter().for_each(|(i, c)| {
        let d = i as DartIdType; // account for the null dart
        let &[b0, b1, b2, b3] = c else { unreachable!() };
        map.set_betas(d, [b0, b1, b2, b3]);
    });

    let vertices = vertices.as_slice()?;
    // let vertices: Vec<_> = vertices
    //     .as_slice()?
    //     .chunks_exact(32)
    //     .flat_map(|c| &c[..24])
    //     .cloned()
    //     .collect();
    // println!("{}", vertices.len());
    // panic!();
    map.par_iter_vertices().for_each(|d| {
        map.force_write_vertex(d as VertexIdType, vertices[d as usize]);
    });
    Ok(map)
}
