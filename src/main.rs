//use std::fs::File;
//use std::io::Write;

use std::time::Instant;

use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

use rayon::prelude::*;

use honeycomb::prelude::{CMap2, CMapBuilder, CoordsFloat, DartIdType};

const N_X: usize = 2048;
const N_Y: usize = 2048;
const N_DARTS: usize = 1 + N_X * N_Y * 4;
const KERNEL: &str = "src/grid.ptx";

const BLOCK_DIMS: (u32, u32, u32) = (4, 4, 4);
const GRID_DIMS: (u32, u32, u32) = (
    (N_X as u32 + BLOCK_DIMS.0 - 1) / BLOCK_DIMS.0,
    (N_Y as u32 + BLOCK_DIMS.1 - 1) / BLOCK_DIMS.1,
    1,
);

fn generate_beta() -> Result<Vec<DartIdType>, DriverError> {
    let launch_params = LaunchConfig {
        grid_dim: GRID_DIMS,
        block_dim: BLOCK_DIMS,
        shared_mem_bytes: 0,
    };

    let mut instant = Instant::now();
    let dev = CudaDevice::new(0)?;
    println!("device acquired in {}ms", instant.elapsed().as_millis());
    instant = Instant::now();

    dev.load_ptx(Ptx::from_file(KERNEL), "grid", &["generate_2d_grid_betaf"])?;
    println!("ptx loaded in {}ms", instant.elapsed().as_millis());
    instant = Instant::now();

    let f = dev.get_func("grid", "generate_2d_grid_betaf").unwrap();

    let mut out_dev = dev.alloc_zeros::<DartIdType>(3 * N_DARTS)?;

    unsafe { f.launch(launch_params, (&mut out_dev, N_X, N_Y, 3 * N_DARTS)) }?;

    let mut out_host: Vec<DartIdType> = vec![0; 3 * N_DARTS];
    dev.dtoh_sync_copy_into(&out_dev, &mut out_host)?;
    println!(
        "grid kernel executed in {}ms",
        instant.elapsed().as_millis()
    );

    #[cfg(debug_assertions)] // debug only
    {
        // we check correctness by building a map on CPU & comparing values
        let map: CMap2<f32> = CMapBuilder::unit_grid(N_X).build().unwrap();
        println!("map built in {}ms", instant.elapsed().as_millis());
        instant = Instant::now();

        let n_correct = out_host
            .chunks(3)
            .enumerate()
            .par_bridge()
            .flat_map(|(i, c)| {
                let d = i as DartIdType; // account for the null dart
                let &[b0, b1, b2] = c else { unreachable!() };
                [
                    b0 as DartIdType == map.beta::<0>(d),
                    b1 as DartIdType == map.beta::<1>(d),
                    b2 as DartIdType == map.beta::<2>(d),
                ]
            })
            .filter(|a| *a)
            .count();
        let n_tot = 3 * N_DARTS;
        assert_eq!(n_tot, n_correct);
        println!("checked validity in {}ms", instant.elapsed().as_millis());
    }

    Ok(out_host)
}

fn build_gpu<T: CoordsFloat>() -> Result<CMap2<T>, DriverError> {
    let betas = generate_beta()?;
    let instant = Instant::now();
    // we check correctness by building a map on CPU & comparing values
    let map: CMap2<T> = CMapBuilder::default().n_darts(N_DARTS).build().unwrap();

    betas.chunks(3).enumerate().par_bridge().for_each(|(i, c)| {
        let d = i as DartIdType; // account for the null dart
        let &[b0, b1, b2] = c else { unreachable!() };
        map.set_betas(d, [b0, b1, b2]);
    });
    println!("[GPU] map built in {}ms", instant.elapsed().as_millis());

    Ok(map)
}

fn main() -> Result<(), DriverError> {
    let map_gpu: CMap2<f32> = build_gpu()?;

    let mut instant = Instant::now();
    let map_cpu: CMap2<f32> = CMapBuilder::unit_grid(N_X).build().unwrap();
    println!("[CPU] map built in {}ms", instant.elapsed().as_millis());

    instant = Instant::now();
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
