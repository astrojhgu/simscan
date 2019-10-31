use fitsimg::write_img;

use clap::{App, Arg};

use ndarray::{ArrayView1, Array1, Array2};
use linear_solver::io::RawMM;
use linear_solver::utils::sp_mul_a1;
use linear_solver::minres::agmres::AGmresState;
use linear_solver::minres::gmres::GmresState;


fn main(){
        let matches=App::new("solve map making problem with noise model")
        .arg(Arg::with_name("pointing matrix")
            .short("p")
            .long("pointing")
            .value_name("pointing_matrix")
            .takes_value(true)
            .help("pointing matrix in matrix market format")
            .required(true)
        )
        .arg(Arg::with_name("tod data")
            .short("t")
            .long("tod")
            .value_name("tod data")
            .takes_value(true)
            .help("tod data")
            .required(true)
        )
        .arg(Arg::with_name("pixel_index")
            .short("x")
            .long("pix")
            .value_name("pixel_index")
            .takes_value(true)
            .help("pixel index")
            .required(true)
        ).get_matches();
        //.arg(Arg::with_name("noise spectrum"))

    let scan=RawMM::<f64>::from_file(matches.value_of("pointing matrix").unwrap()).to_sparse();
    let tod=RawMM::<f64>::from_file(matches.value_of("tod data").unwrap()).to_array1();
    let pix_idx=RawMM::<isize>::from_file(matches.value_of("pixel_index").unwrap()).to_array2();
    println!("{:?}", tod.shape());
    let ata=&scan.transpose_view()*&scan;
    let b=sp_mul_a1(&scan.transpose_view(), tod.view());

    let A = |x: ArrayView1<f64>| -> Array1<f64> {
        //a.dot(&x.to_owned())
        sp_mul_a1(&ata, x)
    };
    
    let M = |x: ArrayView1<f64>| -> Array1<f64> { x.to_owned() };

    let x=Array1::<f64>::zeros(b.len());

    let tol=1e-12;
    //let mut ags = AGmresState::<f64>::new(&A, x.view(), b.view(), &M, 50, 1, 1, 0.4, tol);
    let mut ags = GmresState::<f64>::new(&A, x.view(), b.view(), &M, 30, tol);

    let mut cnt = 0;
    while !ags.converged {
        cnt += 1;
        ags.next(&A, &M);
        //if cnt % 100 == 0 
        {  
            println!("resid={}", ags.resid);
        }
    }

    let (min_i, max_i)={
        let c=pix_idx.column(0);
        (*c.iter().min().unwrap(), *c.iter().max().unwrap())
    };

    let (min_j, max_j)={
        let c=pix_idx.column(1);
        (*c.iter().min().unwrap(), *c.iter().max().unwrap())
    };

    let height=(max_i-min_i+1) as usize;
    let width=(max_j-min_j+1) as usize;

    let mut img=Array2::<f64>::zeros((height, width));

    for (&x, (i, j)) in ags.x.iter().zip(pix_idx.column(0).iter().zip(pix_idx.column(1).iter())){
        img[((i-min_i) as usize, (j-min_j) as usize)]=x;
    }

    write_img("solution.fits".to_string(), &img.into_dyn());
}
