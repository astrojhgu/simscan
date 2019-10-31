use sprs::CsMat;
use ndarray::Array2;
use num_traits::{NumAssign, FloatConst, Float};
use num_complex::Complex;

pub struct BounceScanState{
    dir: (f64, f64),
    point: (f64, f64),
    step_size: f64,
    img_size: f64,
}

pub fn f64mod(x:f64, y:f64)->f64{
    let x1=x%y;
    if x1<0.0{
        x1+y
    }else{
        x1
    }
}

pub fn scan(init_point: (f64, f64), init_dir:(f64, f64), step_size: f64, img_size: f64, nptr: usize)->Vec<(usize, usize)>{
    let  mut s=BounceScanState::new(init_point, init_dir, step_size, img_size);
    (0..nptr).map(|_|{
        s.next_step()
    }).collect()
}

pub fn ij2n(i: usize, j: usize, img_size: usize)->usize{
    i*img_size+j
}

pub fn n2ij(n: usize, img_size: usize)->(usize, usize){
    let i=n/img_size;
    let j=n-i*img_size;
    (i,j)    
}

pub fn get_scan_mat(pixes: &Vec<(usize, usize)>, img_size: usize)->sprs::CsMat<f64>{
    let nscan=pixes.len();
    let iptr:Vec<_>=(0..=nscan).collect();
    let idn:Vec<_>=pixes.iter().map(|&(i,j)|{ij2n(i,j, img_size)}).collect();
    let data=vec![1.0; nscan];
    sprs::CsMat::new((nscan, img_size*img_size), iptr, idn, data)
}

pub fn get_scan_mat_gaussian(pixes: &Vec<(usize, usize)>, img_size: usize, beam_sigma: f64, beam_range: usize)->sprs::CsMat<f64>{
    let nscan=pixes.len();
    let mut iptr=vec![0];
    let mut idn=Vec::new();
    let mut data=Vec::new();
    for &(i0,j0) in pixes.iter(){
        let mut idn1=Vec::new();
        let mut data1=Vec::new();
        for i in (i0 as isize-beam_range as isize..=i0 as isize+beam_range as isize){
            for j in (j0 as isize-beam_range as isize..=j0 as isize+beam_range as isize){
                if i<0 || j<0 || i>=img_size as isize || j>=img_size as isize{
                    continue;
                }
                let r2=((i-i0 as isize) as f64).powi(2)+((j-j0 as isize) as f64).powi(2);
                let b=(-r2/(2.0*beam_sigma)).exp();
                let n=ij2n(i as usize, j as usize, img_size);
                idn1.push(n);
                data1.push(b);
            }
        }
        iptr.push(iptr.last().unwrap()+idn1.len());
        idn.append(&mut idn1);
        data.append(&mut data1);

    }
    sprs::CsMat::new((nscan, img_size*img_size), iptr, idn, data)
}

pub fn sprs2ndarray<T>(input: &sprs::CsMat<T>)->ndarray::Array2<T>
where T: Copy+num_traits::Num{
    let mut output=ndarray::Array2::<T>::zeros((input.rows(), input.cols()));
    for (&x, (i,j)) in input.iter(){
        output[(i,j)]=x;
    }
    output
}

impl BounceScanState{
    pub fn new(current_point: (f64, f64), init_dir: (f64, f64), step_size: f64, img_size: f64)-> BounceScanState{
        BounceScanState{
            dir: init_dir, 
            point: current_point, 
            step_size, 
            img_size
        }
    }

    pub fn regulate1(&self, x: f64)->f64{
        let img_size=self.img_size;

        let x1=f64mod(x,img_size*2.0);

        (if x1>img_size {
            2.0*img_size-x1
        }else{
            x1
        })-0.5
    }


    pub fn regulate (&self, (x,y):(f64, f64))->(f64, f64){
        (self.regulate1(x), self.regulate1(y))
    }

    pub fn next_step(&mut self)->(usize, usize){
        self.point=(f64mod(self.point.0+self.dir.0*self.step_size, self.img_size*2.0),
        f64mod(self.point.1+self.dir.1*self.step_size, self.img_size*2.0));
        let result=self.regulate(self.point);
        return (f64::round(result.0) as usize, f64::round(result.1) as usize)
    }
}

pub fn deconv<T>(data: &[T], kernel: &[Complex<T>])->Vec<T>
where T: Float + FloatConst + NumAssign + std::fmt::Debug
{
    let mut rfft=chfft::RFft1D::<T>::new(data.len());
    let mut s=rfft.forward(data);
    assert!(s.len()==kernel.len());
    for i in 0..s.len(){
        s[i]=s[i]/kernel[i];
    }
    rfft.backward(&s[..])
}

