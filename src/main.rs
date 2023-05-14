use std::path::PathBuf;

use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::PtxCrate,
};

use chrono::prelude::*;

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    // use compile_crate_to_ptx to build and rust kernels in pure rust
    // uses experimental ABI_PTX
    let kernel_path: PathBuf = "kernel/src/lib.rs".into();
    let kernels = PtxCrate::compile_crate_to_ptx(&kernel_path).unwrap();
    let kernel = kernels.first().unwrap();

    println!("loading...");
    dev.load_ptx(kernel.clone(), "rust_kernel", &["thread_id"])?;
    println!("loaded!");

    const GRID: u32 = 32;
    const BLOCK: u32 = 32;
    const N: u32 = GRID * BLOCK;

    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    let cfg = LaunchConfig {
        grid_dim: (GRID, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut thread_enumerator: [i32; N as usize] = [0; N as usize];
    let mut thread_enumerator_dev = dev.htod_copy(thread_enumerator.into())?;
    
    let g = dev.get_func("rust_kernel", "thread_id").unwrap();
    unsafe { g.launch(cfg.clone(), (&mut thread_enumerator_dev, N as i32)) }?;

    let thread_enumerator_results = dev.sync_reclaim(thread_enumerator_dev.clone())?;
    println!("threads enumerated: {thread_enumerator_results:?}");

    let b_host: [u64; N as usize] = [0; N as usize];
    let mut b_dev = dev.htod_copy(b_host.into())?;
    
    
    
    let f = dev.get_func("rust_kernel", "run_length").unwrap();

    let start_time = Local::now();
    println!("start: {start_time:?}");

    unsafe { f.launch(cfg, (&mut b_dev, N as i32)) }?;

    let b_host = dev.sync_reclaim(b_dev.clone())?;
    let goodies: Vec<_> = b_host.into_iter()
        .filter(|b|
    {
        0.ne(b)
    }).map(get_string).collect();
    
    println!("{} goodies: {goodies:?}", goodies.len());
    
    let end_time = Local::now();
    println!("end: {end_time:?}");

    let elapsed = end_time - start_time;
    println!("Time elapsed: {}", elapsed);

    // we can also manage and clean up the build ptx files with a PtxCrate
    let mut rust_ptx: PtxCrate = kernel_path.try_into().unwrap();
    rust_ptx.build_ptx().unwrap();
    let _kernel = rust_ptx.ptx_files().unwrap().first().unwrap();
    println!("cleaned successfully? {:?}", rust_ptx.clean());

    Ok(())
}

fn get_string(seed: u64) -> String {
    const CHARS: [char; 35] = [
        '0', '1', '2', '3', '4',
        '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E',
        'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'P',
        'Q', 'R', 'S', 'T', 'U',
        'V', 'W', 'X', 'Y', 'Z',
    ];    
    const SEED_BASE: u64 = 35;

    let mut seed = seed;
    let mut string = String::new();

    while seed != 0 {
        let rem = (seed % SEED_BASE) as usize;
        seed /= SEED_BASE;
        string.insert(0, CHARS[rem]);
    }
    string
}