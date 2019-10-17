extern crate clap;
extern crate linear_solver;
extern crate ndarray;

use clap::{App, Arg};

use ndarray::{ArrayView1, Array1};
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
        .arg(Arg::with_name("answer")
            .short("a")
            .long("ans")
            .value_name("answer")
            .takes_value(true)
            .help("answer")
            .required(true)
        ).get_matches();
        //.arg(Arg::with_name("noise spectrum"))

    let scan=RawMM::<f64>::from_file(matches.value_of("pointing matrix").unwrap()).to_sparse();
    let tod=RawMM::<f64>::from_file(matches.value_of("tod data").unwrap()).to_array1();
    let answer=RawMM::<f64>::from_file(matches.value_of("answer").unwrap()).to_array1();
    println!("{:?}", tod.shape());
    let ata=&scan.transpose_view()*&scan;
    let b=sp_mul_a1(&scan.transpose_view(), tod.view());
    let b1=sp_mul_a1(&ata, answer.view());

    let r=&b1-&b;

    println!("{:?}", r);
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
        if cnt % 100 == 0 {
            let r=&ags.x-&answer;
            let delta=r.iter().max_by(|a,b|{if a.abs()<b.abs() {
                std::cmp::Ordering::Less
                }else{
                std::cmp::Ordering::Greater
                }
            }).unwrap();

            println!("{} {}", ags.resid, delta);
        }
        ags.next(&A, &M);
    }

    let r=&ags.x-&answer;
    let delta=r.iter().max_by(|a,b|{if a.abs()<b.abs() {
        std::cmp::Ordering::Less
    }else{
        std::cmp::Ordering::Greater
    }    }).unwrap();


    println!("{:?}", &ags.x-&answer);
    println!("{}", delta);
}
