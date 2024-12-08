use std::{io::Result, sync::Arc};

use cudarc::driver::{CudaDevice, CudaSlice, DriverError};
use honeycomb::prelude::{
    CMap2, CMapBuilder, CoordsFloat, DartIdType, Orbit2, OrbitPolicy, Vertex2, VertexIdType,
    NULL_DART_ID,
};

pub type Node = (VertexIdType, Vec<VertexIdType>);

pub fn get_nodes<T: CoordsFloat>(map: &CMap2<T>) -> Vec<Node> {
    map.fetch_vertices()
        .identifiers
        .into_iter()
        .filter_map(|v| {
            if Orbit2::new(&map, OrbitPolicy::Vertex, v as DartIdType)
                .any(|d| map.beta::<2>(d) == NULL_DART_ID)
            {
                None
            } else {
                Some((
                    v,
                    Orbit2::new(&map, OrbitPolicy::Vertex, v as DartIdType)
                        .map(|d| map.vertex_id(map.beta::<2>(d)))
                        .collect(),
                ))
            }
        })
        .collect()
}

#[repr(C)]
#[derive(Debug, Default)]
struct DVertex2<T: CoordsFloat> {
    pub x: T,
    pub y: T,
}

pub fn send_map_to_device<T: CoordsFloat>(
    map: &CMap2<T>,
    dev: &Arc<CudaDevice>,
) -> Result<CudaSlice<DVertex2<T>>, DriverError> {
    let vertices: Vec<DVertex2<T>> = map
        .fetch_vertices()
        .identifiers
        .into_iter()
        .map(|id| {
            if let Some(v) = map.force_read_vertex(id) {
                DVertex2 { x: v.0, y: v.1 }
            } else {
                DVertex2::default()
            }
        })
        .collect();
    dev.htod_copy(vertices)
}
