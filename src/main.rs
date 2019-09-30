extern crate simscan;
extern crate fitsimg;
extern crate ndarray;
extern crate lsqr;

use ndarray::array;

use simscan::{scan, get_scan_mat, get_scan_mat_gaussian};
const IMG_SIZE:usize=16;

fn main() {
    let mut skymap=ndarray::Array2::<f64>::zeros((IMG_SIZE, IMG_SIZE));
    skymap[(8,8)]=1.0;
    //let skymap=array![[1,2],[3,4]];
    println!("{:?}", skymap);
    let skymap_flat=skymap.into_shape((IMG_SIZE*IMG_SIZE,)).unwrap();
    println!("{:?}", skymap_flat);
    
    let ptrs=scan((0., 0.), (1.0, 3.0_f64.sqrt()), 0.1 , IMG_SIZE as f64, 30000);
    let scan_mat=get_scan_mat_gaussian(&ptrs, 16, 1.5, 5);
    let scan_mat_no_deconv=get_scan_mat(&ptrs, 16);

    let scan_mat_solve=&scan_mat;
    let tod=lsqr::sp_mul_a1(&scan_mat, &skymap_flat);
    println!("{}", tod.len());
    let mut lss=lsqr::LsqrState::new(scan_mat_solve, &tod);
    for i in 0..100000{
        lss.next(&scan_mat_solve);
        if i%100 ==0{
            let resid=lss.calc_resid(&scan_mat_solve, &tod);
            println!("{} {}", i, resid.dot(&resid));

        }
    }

    let solution=lss.x.clone().into_shape((IMG_SIZE, IMG_SIZE)).unwrap().into_dyn();

    //println!("{:?}", scan_mat);
    //println!("{} {}" , scan_mat.rows(), scan_mat.cols());
    //let mat=simscan::sprs2ndarray(&scan_mat).into_dyn();
    fitsimg::write_img("a.fits".to_string(), &solution);
}
