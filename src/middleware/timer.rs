extern crate iron;
extern crate time;

use iron::prelude::*;
use iron::{typemap, AfterMiddleware, BeforeMiddleware};
use self::time::precise_time_ns;

pub struct ResponseTime;

impl typemap::Key for ResponseTime { type Value = u64; }

impl BeforeMiddleware for ResponseTime {
    fn before(&self, req: &mut Request) -> IronResult<()> {
        req.extensions.insert::<ResponseTime>(precise_time_ns());
        Ok(())
    }
}

impl AfterMiddleware for ResponseTime {
    fn after(&self, req: &mut Request, res: Response) -> IronResult<Response> {
        let delta = precise_time_ns() - *req.extensions.get::<ResponseTime>().unwrap();

        println!("Request `{}` took: {} ms", req.url, (delta as f64) / 1000000.0);
        Ok(res)
    }
}