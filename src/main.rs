use itertools::Itertools;
use rand::prelude::*;
use std::fmt::Debug;
use std::hash::{DefaultHasher, Hash, Hasher};

type HashValue = u64;

#[derive(Debug)]
enum Node<T: Hash> {
    Node {
        hash: HashValue,
        left: Box<Node<T>>,
        right: Box<Node<T>>,
    },
    Leaf {
        hash: HashValue,
        value: T,
    },
    PaddingLeaf {
        // randomly assigned
        padding: HashValue,
    },
}

#[derive(Debug)]
struct Tree<T: Hash> {
    head: Node<T>,
    len: usize,
    // height includes leaf nodes
    height: usize,
}

// the missing hash at each step of the proof
#[derive(Debug)]
enum ProofComponent {
    Left(HashValue),
    Right(HashValue),
}

#[derive(Debug)]
struct Proof<'a, T: Hash>(&'a T, Vec<ProofComponent>);

impl<T: Hash> Node<T> {
    fn hash_value(&self) -> HashValue {
        match self {
            Node::Node { hash, .. } => *hash,
            Node::Leaf { hash, .. } => *hash,
            Node::PaddingLeaf { padding } => *padding,
        }
    }
}

fn hash_one<T: Hash>(x: &T) -> HashValue {
    let mut hasher = DefaultHasher::new();
    x.hash(&mut hasher);
    hasher.finish()
}

fn hash_two<T: Hash>(x: &T, y: &T) -> HashValue {
    let mut hasher = DefaultHasher::new();
    x.hash(&mut hasher);
    y.hash(&mut hasher);
    hasher.finish()
}

fn bit(num: usize, i: usize) -> bool {
    (num >> i) & 1 == 1
}

impl<T: Hash> Tree<T> {
    fn new<I: IntoIterator<Item = T>>(items: I) -> Self {
        let rng = &mut thread_rng();

        let mut nodes: Vec<_> = items
            .into_iter()
            .map(|x| Node::Leaf {
                hash: hash_one(&x),
                value: x,
            })
            .collect();

        let len = nodes.len();
        assert!(len > 0);

        let mut height = 1;
        while nodes.len() > 1 {
            height += 1;
            nodes = nodes
                .into_iter()
                .chunks(2)
                .into_iter()
                .map(|mut chunk| match (chunk.next(), chunk.next()) {
                    (Some(left), Some(right)) => Node::Node {
                        hash: hash_two(&left.hash_value(), &right.hash_value()),
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    (Some(left), None) => {
                        let right = Node::PaddingLeaf { padding: rng.gen() };
                        Node::Node {
                            hash: hash_two(&left.hash_value(), &right.hash_value()),
                            left: Box::new(left),
                            right: Box::new(right),
                        }
                    }
                    _ => unreachable!(),
                })
                .collect();
        }
        Tree {
            head: nodes.pop().unwrap(),
            len,
            height,
        }
    }

    fn commit(&self) -> (HashValue, usize) {
        let hash = match self.head {
            Node::Node { hash, .. } => hash,
            Node::Leaf { hash, .. } => hash,
            Node::PaddingLeaf { .. } => unreachable!(),
        };
        (hash, self.len)
    }

    fn prove(&self, i: usize) -> Proof<T> {
        assert!(i < self.len);
        // TODO: preallocate
        let mut proof = vec![];

        let mut h = self.height - 1;

        let mut node = &self.head;
        while h > 0 {
            let Node::Node { left, right, .. } = node else {
                unreachable!();
            };
            h -= 1;
            if !bit(i, h) {
                proof.push(ProofComponent::Right(right.hash_value()));
                node = left;
            } else {
                proof.push(ProofComponent::Left(left.hash_value()));
                node = right;
            }
        }
        let Node::Leaf { value, .. } = node else {
            unreachable!()
        };
        Proof(value, proof)
    }
}

impl<T: Hash> Proof<'_, T> {
    fn verify(&self, root: HashValue) -> bool {
        let mut hash = hash_one(self.0);
        for component in self.1.iter().rev() {
            hash = match component {
                ProofComponent::Left(h) => hash_two(h, &hash),
                ProofComponent::Right(h) => hash_two(&hash, h),
            };
        }
        hash == root
    }
}

fn main() {
    let values = [
        "hello world",
        "this is a sentence",
        "third sentence",
        "this is interesting",
        "this will need to be padded",
    ];

    let tree = Tree::new(values);
    println!("{:?}", tree);
    let (root, _) = tree.commit();
    let proof = tree.prove(4);
    println!("{:?}", proof.verify(root));
}
