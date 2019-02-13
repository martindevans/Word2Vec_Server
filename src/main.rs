extern crate word2vec;
extern crate flate2;
extern crate hypernonsense;
extern crate rand;

extern crate iron;
use iron::prelude::*;
use iron::status;
use iron::typemap::Key;

extern crate router;
use router::Router;

extern crate persistent;
use persistent::{ Read };

extern crate clap;
use clap::{ Arg, App };

mod wordvecs;
use wordvecs::vectordict::WordVectorDictionary;

mod middleware;
use middleware::timer::ResponseTime;

#[derive(Copy, Clone)]
pub struct W2VModel;
impl Key for W2VModel { type Value = WordVectorDictionary; }

fn get_vector(req: &mut Request) -> IronResult<Response> {

    //Get W2V model, early exit if one isn't bound
    let model = match req.get::<Read<W2VModel>>() {
        Ok(m) => m,
        Err(_) => return Ok(Response::with((status::InternalServerError, "No Word2Vec Model Loaded")))
    };

    //Get the router from the request so we can inspect the query parameters
    let router = match req.extensions.get::<Router>() {
        Some(r) => r,
        None => return Ok(Response::with((status::InternalServerError, "No Router Found")))
    };

    //Get the word and find in the model
    let query_word = router.find("query_word").unwrap_or("_");
    let vector = model.get_vector(query_word);

    return match vector {
        Some(v) => Ok(Response::with((status::Ok, format!("{:?}", v)))),
        None => Ok(Response::with(status::NotFound))
    };
}

fn get_similar_by_word(req: &mut Request) -> IronResult<Response> {
    
    //Get W2V model, early exit if one isn't bound
    let w2v = match req.get::<Read<W2VModel>>() {
        Ok(m) => m,
        Err(_) => return Ok(Response::with((status::InternalServerError, "No Word2Vec Model Loaded")))
    };

    //Get the router from the request so we can inspect the query parameters
    let router = match req.extensions.get::<Router>() {
        Some(r) => r,
        None => return Ok(Response::with((status::InternalServerError, "No Router Found")))
    };

    //Get the word and find in the model, early exit if not a valid word
    let query_word = router.find("query_word").unwrap_or("_");
    let vector = w2v.get_vector(query_word);
    if !vector.is_some() {
        return Ok(Response::with(status::NotFound));
    }
    let vector = vector.unwrap();

    //Get count of results
    let query_count = std::cmp::min(router.find("query_count").unwrap_or("128").parse::<usize>().unwrap_or(128), 512);

    let nearest = w2v.get_nearest(vector, query_count);

    return Ok(Response::with((status::Ok, format!("{:?}", nearest))));
}

//
//fn get_similar_by_vector(req: &mut Request) -> IronResult<Response> {
//    Ok(Response::with((status::Ok, "todo")))
//}

fn main() {
    
    //Parse command line arguments
    let matches = App::new("Word2Vec Server")
        .version("1.0")
        .author("Martin Evans")
        .about("Serves Word2Vec embeddings")
        .arg(
            Arg::with_name("vectors")
                .short("v")
                .long("vectors")
                .help("Path to load word vectors from")
                .takes_value(true)
                .required(true)
        ).arg(
            Arg::with_name("compressed")
                .short("c")
                .long("compressed")
                .takes_value(false)
                .required(false)
        ).arg(
            Arg::with_name("port")
                .short("p")
                .long("port")
                .takes_value(true)
                .default_value("3000")
                .required(false)
        )
        .get_matches();

    //Parse the port number
    let port = matches.value_of("port").unwrap().parse::<u16>().expect("Failed to parse port number");

    //Load the word vectors
    println!("## Loading Word Vectors...");
    let limit = 250000;
    let model = WordVectorDictionary::create_from_path_limit(
        matches.value_of("vectors").expect("Failed to get vectors from command line args"),
        matches.is_present("compressed"),
        limit
    );
    println!(" - Loaded {:?}!", model.word_count());

    let mut router = Router::new();
    router.get("/get_vector/:query_word", get_vector, "get_vector");
    //router.get("/get_similar/:query_vector", get_similar_by_vector, "get_similar_by_vector");
    router.get("/get_similar/:query_word", get_similar_by_word, "get_similar_by_word");

    //Wrap router in a chain which injects the W2V model
    let mut chain = Chain::new(router);
    chain.link_before(ResponseTime);
    chain.link_before(Read::<W2VModel>::one(model));
    chain.link_after(ResponseTime);

    Iron::new(chain).http(("[::]", port)).unwrap();
}