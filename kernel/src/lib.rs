#![feature(abi_ptx)]        // emitting ptx (unstable)
#![feature(stdsimd)]        // simd instructions (unstable)
#![no_std]                  // CUDA compatibility

use libgdx_xs128::rng::*;

use core::arch::nvptx::*;   // access to thread id, etc

#[panic_handler]
fn my_panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/*
    don't mangle fn name
    array: *mut i32     shared memory for writing output
    size: u32           size of array
*/

#[no_mangle]
pub unsafe extern "ptx-kernel" fn run_length(output: *mut u64, size: i32) {
    /* https://doc.rust-lang.org/stable/core/arch/nvptx/index.html */
    let thread_id: i32 = _thread_idx_x();
    let block_id: i32 = _block_idx_x();
    
    let block_dim: i32 = _block_dim_x();
    let grid_dim: i32 = _grid_dim_x();
    
    let n_threads = (block_dim * grid_dim) as u64;
    
    let thread_index = 
        thread_id + 
        block_id * block_dim
    ;

    if thread_index.ge(&size) {
        return
    }
    
    let mut seed = thread_id as u64 + 1;
    while seed < 1_000_000_000_000 {
        let mut rng = Random::new(seed);
        let first = rng.next_capped_u64(71);
        let run: u32 = (0..).find(|i| {
            rng.next_capped_u64(71) != first
        }).unwrap();
        if run >= 5 {
            *output.offset(thread_id as isize) = seed;
            return
        }
        seed = seed.wrapping_add(n_threads)
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn thread_id(output: *mut i32, size: i32) {
    /* https://doc.rust-lang.org/stable/core/arch/nvptx/index.html */
    let thread_id: i32 = _thread_idx_x();
    let block_id: i32 = _block_idx_x();
    
    let block_dim: i32 = _block_dim_x();
    let grid_dim: i32 = _grid_dim_x();
    
    let n_threads = (block_dim * grid_dim) as u64;
    
    let thread_index = 
        thread_id + 
        block_id * block_dim
    ;

    if thread_index.ge(&size) {
        return
    } else {
        *output.offset(thread_index as isize) = thread_index;
    }
}
