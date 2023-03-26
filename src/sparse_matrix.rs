use nalgebra::*;
use nalgebra_sparse::csr::CsrMatrix;
use rayon::prelude::*;
use sprs::{CsMat, CsVec};
use std::time::{Duration, Instant};
extern crate num_cpus;

#[cfg(test)]
mod sparse_matrix_tests {
    use super::*;
    #[test]
    fn sp_mv() {
        let nx = 10;
        let ny = 10;

        let coo = build_sp_matrix_coo(nx, ny);

        let a = convert_sp_matrix_from_coo_to_csr(coo);
        let b = build_vector(a.size);

        let (c0, _) = sp_mv_mycsr_for(&a, &b);

        let (c1, _) = sp_mv_mycsr_par_iter(&a, &b);
        assert_eq!(c0, c1);
        let (c1, _) = sp_mv_mycsr_par_iter2(&a, &b);
        assert_eq!(c0, c1);

        let (c1, _) = sp_mv_mycsr_iter(&a, &b);
        assert_eq!(c0, c1);

        let (c1, _) = sp_mv_mycsr_iter2(&a, &b);
        assert_eq!(c0, c1);

        assert_eq!(c0, c1);
        let (c1, _) = sp_mv_nalgebra(&a, &b);
        assert_eq!(c0, c1);
        let (c1, _) = sp_mv_sprs(&a, &b);
        assert_eq!(c0, c1);
    }
    #[test]
    fn build_spmatrix() {
        let nx = 3;
        let ny = 4;
        //   0 1 2
        // 0 x x x
        // 1 x x x
        // 2 x x x
        // 3 x x x

        let coo = build_sp_matrix_coo(nx, ny);
        // 非ゼロ要素は隣接5格子*12格子-境界(3+3+4+4)要素=46
        assert_eq!(coo.col_index.len(), 46);
        assert_eq!(coo.row_index.len(), 46);
        assert_eq!(coo.val.len(), 46);

        let a = convert_sp_matrix_from_coo_to_csr(coo);
        // row_prtは行列の行数+1
        assert_eq!(a.col_index.len(), 46);
        assert_eq!(a.row_ptr.len(), 13);
        assert_eq!(a.val.len(), 46);

        let b = build_vector(a.size);

        let (c0, _) = sp_mv_mycsr_for(&a, &b);
        assert_eq!(
            c0,
            [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]
        );
    }
}

struct SparseMatrixCoo {
    val: Vec<f64>,
    row_index: Vec<usize>,
    col_index: Vec<usize>,
    size: usize,
}

struct SparseMatrixCsr {
    val: Vec<f64>,
    row_ptr: Vec<usize>,
    col_index: Vec<usize>,
    size: usize,
}

fn build_sp_matrix_coo(nx: usize, ny: usize) -> SparseMatrixCoo {
    let n = nx * ny;
    let mut val = vec![0.0; 0];
    let mut row_index = vec![0; 0];
    let mut col_index = vec![0; 0];

    let dx = 1.0;
    let nx = nx as isize;
    let n = n as isize;

    for i in 0..n {
        // jは i-nx,i-1, i ,i+1,  i+nx
        let mut j_cand = vec![0; 0];
        if 0 <= i - nx && i - nx < n {
            j_cand.push(i - nx);
        }

        // x_plus (i - 1)
        if i - 1 < n && (i % nx != 0) {
            //0 <= i - 1 &&は不要になるので省いた
            j_cand.push(i - 1);
        }

        // diag (i)
        if 0 <= i && i < n {
            j_cand.push(i);
        }

        // x_minus (i + 1)
        if 0 <= i + 1 && i + 1 < n && ((i + 1) % nx != 0) {
            j_cand.push(i + 1);
        }
        // y_minus (i + nx)
        if 0 <= i + nx && i + nx < n {
            j_cand.push(i + nx);
        }

        for j in j_cand.iter() {
            if *j == i {
                val.push(-((j_cand.len() - 1) as f64) / (dx * dx));
            } else {
                val.push(1.0 / (dx * dx));
            }
            col_index.push(*j as usize);
            row_index.push(i as usize);
        }
    }

    SparseMatrixCoo {
        size: n as usize,
        val,
        row_index,
        col_index,
    }
}

#[allow(dead_code)]
fn print_sp_matrix_coo(mat: &SparseMatrixCoo) {
    let mut ptr = 0;
    for i in 0..mat.size {
        let mut sentence = "".to_string();
        for j in 0..mat.size {
            if ptr > mat.val.len() {
                sentence += &(0.0).to_string();
            } else if mat.row_index[ptr] == i && mat.col_index[ptr] == j {
                sentence += &mat.val[ptr].to_string();
                ptr += 1;
            } else {
                sentence += &(0.0).to_string();
            }
        }
        println!("{}", sentence);
    }
}

fn convert_sp_matrix_from_coo_to_csr(mat: SparseMatrixCoo) -> SparseMatrixCsr {
    let mut row_ptr = vec![0; mat.size + 1];

    for i in 0..mat.val.len() {
        row_ptr[(mat.row_index[i] + 1)] += 1;
    }

    for i in 0..mat.size {
        row_ptr[i + 1] += row_ptr[i];
    }

    SparseMatrixCsr {
        val: mat.val,
        row_ptr,
        col_index: mat.col_index,
        size: mat.size,
    }
}

fn build_vector(n: usize) -> Vec<f64> {
    let b = vec![0.0; n];
    b.into_iter()
        .enumerate()
        .map(|(i, _vij)| (i + 1) as f64)
        .collect()
}

fn sp_mv_mycsr_for(mat: &SparseMatrixCsr, b: &[f64]) -> (Vec<f64>, Duration) {
    let n = mat.size;
    let t0 = Instant::now();
    let mut c = vec![0.0; n];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        for j in mat.row_ptr[i]..mat.row_ptr[i + 1] {
            c[i] += mat.val[j] * b[mat.col_index[j]];
        }
    }
    let t1 = t0.elapsed();
    (c, t1)
}

#[allow(dead_code)]
fn sp_mv_mycsr_par_iter(mat: &SparseMatrixCsr, b: &[f64]) -> (Vec<f64>, Duration) {
    let num_thread = num_cpus::get();
    let n = mat.size;
    let t0 = Instant::now();
    let c = (0..n)
        .into_par_iter()
        .with_min_len(n / num_thread)
        .map(|i| {
            (mat.row_ptr[i]..mat.row_ptr[i + 1])
                .map(|j| mat.val[j] * b[mat.col_index[j]])
                .sum()
        })
        .collect();
    let t1 = t0.elapsed();
    (c, t1)
}

#[allow(dead_code)]
fn sp_mv_mycsr_par_iter2(mat: &SparseMatrixCsr, b: &[f64]) -> (Vec<f64>, Duration) {
    let num_thread = num_cpus::get();
    let n = mat.size;
    let t0 = Instant::now();
    let mut c = vec![0.0; n];
    c.par_iter_mut()
        .with_min_len(n / num_thread)
        .enumerate()
        .for_each(|(i, ci)| {
            *ci = (mat.row_ptr[i]..mat.row_ptr[i + 1])
                .map(|j| mat.val[j] * b[mat.col_index[j]])
                .sum()
        });
    let t1 = t0.elapsed();
    (c, t1)
}

#[allow(dead_code)]
fn sp_mv_mycsr_iter(mat: &SparseMatrixCsr, b: &[f64]) -> (Vec<f64>, Duration) {
    let n = mat.size;
    let t0 = Instant::now();

    #[allow(clippy::useless_conversion)]
    let c = (0..n)
        .into_iter()
        .map(|i| {
            (mat.row_ptr[i]..mat.row_ptr[i + 1])
                .map(|j| mat.val[j] * b[mat.col_index[j]])
                .sum()
        })
        .collect();
    let t1 = t0.elapsed();
    (c, t1)
}

#[allow(dead_code)]
fn sp_mv_mycsr_iter2(mat: &SparseMatrixCsr, b: &[f64]) -> (Vec<f64>, Duration) {
    let n = mat.size;
    let t0 = Instant::now();
    let mut c = vec![0.0; n];
    c.iter_mut().enumerate().for_each(|(i, ci)| {
        *ci = (mat.row_ptr[i]..mat.row_ptr[i + 1])
            .map(|j| mat.val[j] * b[mat.col_index[j]])
            .sum()
    });
    let t1 = t0.elapsed();
    (c, t1)
}

#[allow(dead_code)]
fn sp_mv_sprs(mat: &SparseMatrixCsr, b: &[f64]) -> (Vec<f64>, Duration) {
    let n = mat.size;
    let mut ind = vec![0; n];
    for (i, indi) in ind.iter_mut().enumerate() {
        *indi = i
    }
    let b_sp = CsVec::new(n, ind, b.to_vec());
    let a_sp = CsMat::new(
        (n, n),
        mat.row_ptr.clone(),
        mat.col_index.clone(),
        mat.val.clone(),
    )
    .to_csc();

    let t0 = Instant::now();
    let c = &a_sp * &b_sp;
    let t1 = t0.elapsed();
    (c.data().to_vec(), t1)
}

#[allow(dead_code)]
fn sp_mv_nalgebra(mat: &SparseMatrixCsr, b: &[f64]) -> (Vec<f64>, Duration) {
    let n = mat.size;
    let a_na = CsrMatrix::try_from_csr_data(
        n,
        n,
        mat.row_ptr.clone(),
        mat.col_index.clone(),
        mat.val.clone(),
    )
    .unwrap();

    let b_na = DVector::from_column_slice(b);

    let t0 = Instant::now();
    let c = &a_na * &b_na;
    let t1 = t0.elapsed();

    let mut c2 = vec![0.0; n];
    for i in 0..c.len() {
        c2[i] = c[i];
    }
    (c2, t1)
}

fn measure_average_time<F>(a: &SparseMatrixCsr, b: &[f64], f: F) -> Duration
where
    F: Fn(&SparseMatrixCsr, &[f64]) -> (Vec<f64>, Duration),
{
    let iter_num = 10;
    let mut time = Duration::new(0, 0);
    for _ in 0..iter_num {
        let (_, t1) = f(a, b);
        time += t1;
    }
    time / iter_num
}

pub fn sparse_matvecmul(nx: usize, ny: usize) {
    println!(
        "==================================matrix_size:{},{}",
        nx, ny
    );

    let coo = build_sp_matrix_coo(nx, ny);
    // print_sp_matrix_coo(&coo);
    let a = convert_sp_matrix_from_coo_to_csr(coo);
    let b = build_vector(a.size);
    // println!("{:?}", a.val);
    // println!("{:?}", a.col_index);
    // println!("{:?}", a.row_ptr);
    //base answer
    let t = measure_average_time(&a, &b, sp_mv_mycsr_for);
    println!("sp_mv_mycsr_for:\t {:?} μs", t.as_micros());

    let t = measure_average_time(&a, &b, sp_mv_mycsr_iter);
    println!("sp_mv_mycsr_iter:\t {:?} μs", t.as_micros());
    let t = measure_average_time(&a, &b, sp_mv_mycsr_iter2);
    println!("sp_mv_mycsr_iter2:\t {:?} μs", t.as_micros());

    let t = measure_average_time(&a, &b, sp_mv_mycsr_par_iter);
    println!("sp_mv_mycsr_par_iter:\t {:?} μs", t.as_micros());
    let t = measure_average_time(&a, &b, sp_mv_mycsr_par_iter2);
    println!("sp_mv_mycsr_par_iter2:\t {:?} μs", t.as_micros());

    let t = measure_average_time(&a, &b, sp_mv_sprs);
    println!("sp_mv_sprs:\t\t {:?} μs", t.as_micros());
    let t = measure_average_time(&a, &b, sp_mv_nalgebra);
    println!("sp_mv_nalgebla:\t\t {:?} μs", t.as_micros());
}
