mod dim2;
mod dim3;

use std::time::Instant;

use cudarc::driver::DriverError;
use honeycomb::prelude::{grid_generation::GridBuilder, CMap2, CMap3, DartIdType};
use rayon::prelude::*;

fn main() -> Result<(), DriverError> {
    run_3d()?;
    Ok(())
}

pub fn run_3d() -> Result<(), DriverError> {
    // generate using a GPU -- scopes allow maps to drop and free memory
    // uncomment them and comment out the verification for larger cases
    // {
    // let instant = Instant::now();
    // let map_gpu: CMap3<f32> = dim3::build_gpu()?;
    // println!("[GPU] map built in {}ms", instant.elapsed().as_millis());
    // }
    // generate using the CPU
    // {
    let instant = Instant::now();
    let map_cpu: CMap3<f32> = GridBuilder::default()
        .n_cells([dim3::N_X, dim3::N_Y, dim3::N_Z])
        .len_per_cell([1.0, 1.0, 1.0])
        .build()
        .unwrap();
    println!("[CPU] map built in {}ms", instant.elapsed().as_millis());
    // }

    // check consistency
    // let instant = Instant::now();
    // assert_eq!(map_cpu.n_darts(), map_gpu.n_darts());
    // let n_correct = (0..dim3::N_DARTS as DartIdType)
    //     .into_par_iter()
    //     .flat_map(|d| {
    //         [
    //             (0, map_gpu.beta::<0>(d) == map_cpu.beta::<0>(d)),
    //             (1, map_gpu.beta::<1>(d) == map_cpu.beta::<1>(d)),
    //             (2, map_gpu.beta::<2>(d) == map_cpu.beta::<2>(d)),
    //             (3, map_gpu.beta::<3>(d) == map_cpu.beta::<3>(d)),
    //         ]
    //     })
    //     .filter(|(i, a)| {
    //         if !*a {
    //             println!("b{i} false")
    //         }
    //         *a
    //     })
    //     .count();
    // let n_tot = 4 * dim3::N_DARTS;
    // assert_eq!(n_tot, n_correct);
    // println!("checked result in {}ms", instant.elapsed().as_millis());

    Ok(())
}

pub fn run_2d() -> Result<(), DriverError> {
    // generate using a GPU -- scopes allow maps to drop and free memory
    // uncomment them and comment out the verification for larger cases
    // {
    let instant = Instant::now();
    let map_gpu: CMap2<f32> = dim2::build_gpu()?;
    println!("[GPU] map built in {}ms", instant.elapsed().as_millis());
    // }
    // generate using the CPU
    // {
    let instant = Instant::now();
    let map_cpu: CMap2<f32> = GridBuilder::default()
        .n_cells([dim2::N_X, dim2::N_Y])
        .len_per_cell([1.0, 1.0])
        .build()
        .unwrap();
    println!("[CPU] map built in {}ms", instant.elapsed().as_millis());
    // }

    // check consistency
    let instant = Instant::now();
    assert_eq!(map_cpu.n_darts(), map_gpu.n_darts());
    let n_correct = (0..dim2::N_DARTS as DartIdType)
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
    let n_tot = 3 * dim2::N_DARTS;
    assert_eq!(n_tot, n_correct);
    println!("checked result in {}ms", instant.elapsed().as_millis());

    Ok(())
}
