use std::io::BufRead;
use std::io::BufReader;
use std::fs::File;
use std::collections::HashMap;

use rand::prelude::*;

use flate2::read::GzDecoder;

use word2vec::vectorreader::WordVectorReader;

use hypernonsense::multiindex::{ MultiIndex };

pub struct WordVectorDictionary {
    word_to_id: HashMap<String, usize>,
    words: Vec<String>,
    vectors: Vec<Vec<f32>>,
    index: MultiIndex<usize>,
    dim: usize
}

impl WordVectorDictionary {

    pub fn create_from_path_limit(path: &str, is_compressed: bool, limit: usize) -> WordVectorDictionary {
        let file = File::open(path).unwrap();
        let buf_reader = BufReader::new(file);

        if is_compressed {
            return WordVectorDictionary::create_from_reader_limit(BufReader::new(GzDecoder::new(buf_reader)), limit);
        } else {
            return WordVectorDictionary::create_from_reader_limit(buf_reader, limit);
        }
    }

    fn create_from_reader_limit<R : BufRead>(file_reader: R, limit: usize) -> WordVectorDictionary {

        let vector_reader = WordVectorReader::new_from_reader(file_reader).unwrap();
        let vsize = vector_reader.vector_size();

        fn normalize(item: (String, Vec<f32>)) -> (String, Vec<f32>) {
            let mut vector = item.1;
            let norm = 1.0 / vector.iter().fold(0f32, |sum, &x| sum + (x * x)).sqrt();
            for x in vector.iter_mut() {
                (*x) *= norm;
            }
            return (item.0, vector);
        }

        let mut result = WordVectorDictionary {
            word_to_id: HashMap::with_capacity(limit),
            vectors: Vec::with_capacity(limit),
            words: Vec::with_capacity(limit),
            index: MultiIndex::new(vsize, 25, 20, &mut thread_rng()),
            dim: vsize
        };

        for (word, vector) in vector_reader.into_iter().take(limit).map(normalize) {
            let id = result.vectors.len();
            result.vectors.push(vector.clone());
            result.words.push(word.clone());
            result.word_to_id.insert(word, id);
            result.index.add(id, &vector);
        }

        return result;
    }

    pub fn word_dimension(&self) -> usize {
        return self.dim;
    }

    pub fn word_count(&self) -> usize {
        return self.vectors.len();
    }

    pub fn get_vector(&self, key: &str) -> Option<&Vec<f32>> {
        return self
            .word_to_id.get(key)
            .map(|id| &self.vectors[*id]);
    }

    pub fn get_nearest(&self, vector: &Vec<f32>, count: usize) -> Vec<(f32, &String)> {

        pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
            assert!(a.len() == b.len());

            let mut acc = 0f32;
            for index in 0..a.len() {
                acc += a[index] * b[index];
            }

            return 2f32 - (acc + 1f32).max(0f32);
        }

        return self.index.nearest(vector, count, |p, k| {
            return cosine_distance(p, &self.vectors[*k]);
        }).iter().map(|a| {
            return (a.distance, &self.words[a.key]);
        }).collect();
    }
}