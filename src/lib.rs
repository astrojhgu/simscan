extern crate ndarray;
extern crate sprs;


use ndarray::Array2;


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

pub fn gen_tod(){

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

