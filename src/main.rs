//use std::fs::File;
//use std::io::Write;

use std::{sync::Arc, time::Instant};

use cudarc::{
    driver::{CudaDevice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig, ValidAsZeroBits},
    nvrtc::Ptx,
};

use rayon::prelude::*;

use honeycomb::prelude::{CMap2, CMapBuilder, CoordsFloat, Vertex2};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

const N_X: usize = 2048;
const N_Y: usize = 2048;
const LEN_CELL_X: f32 = 0.0;
const LEN_CELL_Y: f32 = 0.0;
const N_DARTS: usize = 1 + N_X * N_Y * 4;
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

fn generate_beta(dev: Arc<CudaDevice>) -> Result<Vec<DartIdType>, DriverError> {
    let launch_params = LaunchConfig {
        grid_dim: GRID_DIMS,
        block_dim: BLOCK_DIMS,
        shared_mem_bytes: 0,
    };
    let instant = Instant::now();

    let f = dev.get_func("grid", "generate_2d_grid_betaf").unwrap();
    let mut out_dev = dev.alloc_zeros::<DartIdType>(3 * N_DARTS)?;

    unsafe { f.launch(launch_params, (&mut out_dev, N_X, N_Y, 3 * N_DARTS)) }?;

    let mut out_host: Vec<DartIdType> = vec![0; 3 * N_DARTS];
    dev.dtoh_sync_copy_into(&out_dev, &mut out_host)?;

    println!(
        "grid kernel executed in {}ms",
        instant.elapsed().as_millis()
    );

    Ok(out_host)
}

fn generate_vertices(dev: Arc<CudaDevice>) -> Result<Vec<CuVertex2>, DriverError> {
    let launch_params = LaunchConfig {
        grid_dim: GRID_DIMS,
        block_dim: BLOCK_DIMS,
        shared_mem_bytes: 0,
    };
    let instant = Instant::now();

    let f = dev.get_func("grid", "generate_2d_grid_vertices").unwrap();
    let mut out_dev = dev.alloc_zeros::<CuVertex2>(N_DARTS)?;

    unsafe {
        f.launch(
            launch_params,
            (&mut out_dev, LEN_CELL_X, LEN_CELL_Y, N_X, N_Y, N_DARTS),
        )
    }?;

    let mut out_host: Vec<CuVertex2> = vec![CuVertex2::default(); N_DARTS];
    dev.dtoh_sync_copy_into(&out_dev, &mut out_host)?;

    println!(
        "grid kernel executed in {}ms",
        instant.elapsed().as_millis()
    );

    Ok(out_host)
}

fn build_gpu<T: CoordsFloat>() -> Result<CMap2<T>, DriverError> {
    let dev = CudaDevice::new(0)?;
    let kernel = Ptx::from_src(KERNEL);
    dev.load_ptx(
        kernel,
        "grid",
        &["generate_2d_grid_betaf", "generate_2d_grid_vertices"],
    )?;

    // N_DARTS-1 bc the constant count the null dart; the builder does too
    let map: CMap2<T> = CMapBuilder::default().n_darts(N_DARTS - 1).build().unwrap();

    let betas = generate_beta(dev.clone())?;
    let bcs = betas.chunks(3).enumerate().collect::<Vec<_>>();
    bcs.par_iter().for_each(|(i, c)| {
        let d = *i as DartIdType; // account for the null dart
        let [b0, b1, b2] = c else { unreachable!() };
        map.set_betas(d, [*b0, *b1, *b2]);
    });

    let vertices = generate_vertices(dev.clone())?;
    let vids = (1..map.n_darts() as DartIdType)
        .zip(vertices.into_iter())
        .filter(|(d, _)| *d as VertexIdType == map.vertex_id(*d))
        .collect::<Vec<_>>();
    vids.par_iter().for_each(|(d, v)| {
        map.force_write_vertex(*d as VertexIdType, *v);
    });
    Ok(map)
}

fn main() -> Result<(), DriverError> {
    // generate using a GPU
    let mut instant = Instant::now();
    let map_gpu: CMap2<f32> = build_gpu()?;
    println!("[GPU] map built in {}ms", instant.elapsed().as_millis());

    // generate using the CPU
    instant = Instant::now();
    let map_cpu: CMap2<f32> = CMapBuilder::unit_grid(N_X).build().unwrap();
    println!("[CPU] map built in {}ms", instant.elapsed().as_millis());

    // check consistency
    instant = Instant::now();
    assert_eq!(map_cpu.n_darts(), map_gpu.n_darts());
    let n_correct = (0..N_DARTS as DartIdType)
        .into_par_iter()
        .flat_map(|d| {
            [
                map_gpu.beta::<0>(d) == map_cpu.beta::<0>(d),
                map_gpu.beta::<1>(d) == map_cpu.beta::<1>(d),
                map_gpu.beta::<2>(d) == map_cpu.beta::<2>(d),
            ]
        })
        .filter(|a| *a)
        .count();
    let n_tot = 3 * N_DARTS;
    assert_eq!(n_tot, n_correct);
    println!("checked result in {}ms", instant.elapsed().as_millis());

    Ok(())
}
