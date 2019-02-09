use std::io::BufRead;
use std::io::BufReader;
use std::fs::File;

use flate2::read::GzDecoder;

use word2vec::vectorreader::WordVectorReader;

use kdtree::KdTree;

pub struct WordVectorDictionary {
    map: std::collections::HashMap<String, Vec<f32>>,
    tree: kdtree::KdTree<f32, String, Vec<f32>>,
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
            map: vector_reader.into_iter().take(limit).map(normalize).collect(),
            dim: vsize,
            tree: KdTree::new_with_capacity(1, 2)
        };

        result.tree = KdTree::new_with_capacity(result.word_dimension(), result.word_count());
        for (word, vector) in &result.map {
            result.tree.add(vector.clone(), word.clone()).expect("Failed to add item to KD Tree");
        }

        return result;
    }

    pub fn word_dimension(&self) -> usize {
        return self.dim;
    }

    pub fn word_count(&self) -> usize {
        return self.map.len();
    }

    pub fn get_vector(&self, key: &str) -> Option<&Vec<f32>> {
        return self.map.get(key);
    }

    pub fn get_nearest(&self, vector: &Vec<f32>, count: usize) -> Result<Vec<(f32, &String)>, kdtree::ErrorKind> {

        pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
            assert!(a.len() == b.len());

            let mut acc = 0f32;
            for index in 0..a.len() {
                acc += a[index] * b[index];
            }

            return 2f32 - (acc + 1f32);
        }

        return self.tree.nearest(&vector, count, &cosine_distance);
    }
}