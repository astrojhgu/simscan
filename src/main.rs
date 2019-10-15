extern crate simscan;
extern crate fitsimg;
extern crate ndarray;
extern crate linear_solver;

use linear_solver::io::RawMM;
use ndarray::array;

use simscan::{scan, get_scan_mat, get_scan_mat_gaussian, sprs2ndarray};
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

    let ata=&(scan_mat.transpose_view().to_owned())*&scan_mat;
    let ata_no_deconv=&(scan_mat_no_deconv.transpose_view().to_owned())*&scan_mat_no_deconv;

    let tod=linear_solver::utils::sp_mul_a1(&scan_mat, skymap_flat.view());
    fitsimg::write_img("scan.fits".to_string(), &sprs2ndarray(&scan_mat).into_dyn());
    fitsimg::write_img("scan_no_conv.fits".to_string(), &sprs2ndarray(&scan_mat_no_deconv).into_dyn());
    fitsimg::write_img("ata.fits".to_string(), &sprs2ndarray(&ata).into_dyn());
    fitsimg::write_img("ata_no_conv.fits".to_string(), &sprs2ndarray(&ata_no_deconv).into_dyn());

    let mm=RawMM::from_sparse(&scan_mat);
    mm.to_file("A.mtx");
    let mm=RawMM::from_array1(skymap_flat.view());
    mm.to_file("tod.mtx");
}
