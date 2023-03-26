mod sparse_matrix;

fn main() {
    sparse_matrix::sparse_matvecmul(3, 3);
    sparse_matrix::sparse_matvecmul(10, 10);
    sparse_matrix::sparse_matvecmul(20, 20);
    sparse_matrix::sparse_matvecmul(50, 50);
    sparse_matrix::sparse_matvecmul(100, 100);
    sparse_matrix::sparse_matvecmul(200, 200);
    sparse_matrix::sparse_matvecmul(500, 500);
    sparse_matrix::sparse_matvecmul(1000, 1000);
    sparse_matrix::sparse_matvecmul(2000, 2000);
    sparse_matrix::sparse_matvecmul(5000, 5000);
}
