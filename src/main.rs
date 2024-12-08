use std::intrinsics::atomic_cxchgweak_acquire_seqcst;

use cudarc::driver::ValidAsZeroBits;
use honeycomb::core::stm::atomically;
use honeycomb::kernels::coloring::{color_dsatur, Color};
use honeycomb::prelude::{
    CMap2, CMapBuilder, DartIdType, Orbit2, OrbitPolicy, Vertex2, VertexIdType, NULL_DART_ID,
};
use poc_honeycomb_gpu::{get_nodes, Node};

fn main() {
    // ./binary grid_size n_rounds
    let args: Vec<String> = std::env::args().collect();
    let n_squares = args
        .get(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256);
    let n_rounds = args
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100);

    let mut map: CMap2<f64> = CMapBuilder::unit_grid(n_squares).build().unwrap();
    let colors = 0..=color_dsatur(&mut map);

    // fetch all vertices that are not on the boundary of the map
    // this should prolly be yeeted to device as is? along w/ vertices values
    let nodes: Vec<Node> = get_nodes(&map);

    // throw these incrementally as args of a kernel?
    let sets: Vec<Vec<VertexIdType>> = colors
        .map(|c| {
            nodes
                .iter()
                .filter_map(|(v, _)| {
                    if map.force_read_attribute(*v) == Some(Color(c)) {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();

    // main loop
    let mut round = 0;
    loop {
        nodes.iter().for_each(|(vid, neigh)| {
            atomically(|trans| {
                let mut new_val = Vertex2::default();
                for v in neigh {
                    let vertex = map.read_vertex(trans, *v)?.unwrap();
                    new_val.0 += vertex.0;
                    new_val.1 += vertex.1;
                }
                new_val.0 /= neigh.len() as f64;
                new_val.1 /= neigh.len() as f64;
                map.write_vertex(trans, *vid, new_val)
            });
        });

        round += 1;
        if round >= n_rounds {
            break;
        }
    }

    std::hint::black_box(map);
}
