mod dim2;
mod dim3;

use std::time::Instant;

use cudarc::driver::DriverError;
use honeycomb::prelude::{grid_generation::GridBuilder, CMap2, DartIdType};
use rayon::prelude::*;

use crate::dim2::{build_gpu, N_DARTS, N_X, N_Y};

fn main() -> Result<(), DriverError> {
    // generate using a GPU -- scopes allow maps to drop and free memory
    // uncomment them and comment out the verification for larger cases
    // {
    let instant = Instant::now();
    let map_gpu: CMap2<f32> = build_gpu()?;
    println!("[GPU] map built in {}ms", instant.elapsed().as_millis());
    // }
    // generate using the CPU
    // {
    let instant = Instant::now();
    let map_cpu: CMap2<f32> = GridBuilder::default()
        .n_cells([N_X, N_Y])
        .len_per_cell([1.0, 1.0])
        .build()
        .unwrap();
    println!("[CPU] map built in {}ms", instant.elapsed().as_millis());
    // }

    // check consistency
    let instant = Instant::now();
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
