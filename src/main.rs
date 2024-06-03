#![feature(iter_collect_into)]
#![feature(new_uninit)]

use rand::prelude::*;
use std::fmt::Debug;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::mem::MaybeUninit;
use std::process::abort;
use std::ptr;

type HashValue = u64;

#[derive(Debug)]
struct NodeLinks<T: Hash> {
    left: Node<T>,
    right: Node<T>,
}

#[derive(Debug)]
enum Node<T: Hash> {
    Node {
        hash: HashValue,
        links: Box<NodeLinks<T>>,
    },
    Leaf {
        hash: HashValue,
        value: Box<T>,
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
    fn new<I: IntoIterator<Item = T>>(items: I) -> Self
    where
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
    {
        let rng = &mut thread_rng();

        let items = items.into_iter();
        let mut nodes = Box::new_uninit_slice(items.len());

        // fill the entire slice
        items.enumerate().for_each(|(i, x)| {
            nodes[i].write(Node::Leaf {
                hash: hash_one(&x),
                value: Box::new(x),
            });
        });

        let len = nodes.len();
        assert!(len > 0);

        // number of actually initialized elements in `nodes`
        let mut nodes_len = nodes.len();
        let mut height = 1;
        while nodes_len > 1 {
            height += 1;
            (0..nodes_len).step_by(2).for_each(|i| {
                let left = unsafe {
                    let mut left = MaybeUninit::uninit();
                    ptr::copy_nonoverlapping(&nodes[i], &mut left, 1);
                    left.assume_init()
                };
                let right = nodes
                    .get(i + 1)
                    .map(|m| unsafe {
                        let mut right = MaybeUninit::uninit();
                        ptr::copy_nonoverlapping(m, &mut right, 1);
                        right.assume_init()
                    })
                    .unwrap_or_else(|| Node::PaddingLeaf { padding: rng.gen() });
                nodes[i / 2].write(Node::Node {
                    hash: hash_two(&left.hash_value(), &right.hash_value()),
                    links: Box::new(NodeLinks { left, right }),
                });
            });
            nodes_len = (nodes_len + 1) / 2;
        }
        Tree {
            head: unsafe {
                let mut head = MaybeUninit::uninit();
                ptr::copy_nonoverlapping(&nodes[0], &mut head, 1);
                head.assume_init()
            },
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
            let Node::Node { links, .. } = node else {
                unreachable!();
            };
            h -= 1;
            if !bit(i, h) {
                proof.push(ProofComponent::Right(links.right.hash_value()));
                node = &links.left;
            } else {
                proof.push(ProofComponent::Left(links.left.hash_value()));
                node = &links.right;
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

#[test]
fn test_proof() {
    let values = ["a", "b", "c", "d", "e", "f", "g", "h"];
    for len in 1..values.len() {
        let tree = Tree::new(&values[..len]);
        let (root, _) = tree.commit();
        for i in 0..len {
            let proof = tree.prove(i);
            assert!(proof.verify(root));
        }
    }
}

fn main() {
    use std::time::Instant;

    let start = Instant::now();
    let tree = Tree::new(0..100_000_000);
    println!("tree creation: {:?}", start.elapsed());
    println!("height: {:?}", tree.height);

    let (root, _) = tree.commit();

    let start = Instant::now();
    let proof = tree.prove(50_000_000);
    println!("proof creation: {:?}", start.elapsed());
    println!("proof len: {}", proof.1.len());

    let start = Instant::now();
    println!("{:?}", proof.verify(root));
    println!("proof verification: {:?}", start.elapsed());

    abort()
}
