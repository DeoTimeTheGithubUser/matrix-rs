# matrix-rs
Library for using matrices in rust. Uses const generics to ensure compile-time matrix safety.

### Examples:
```rust
use matrix_rs::*;

fn matrix_3x3_1to9() -> Matrix<3, 3> {
    matrix! {
        1 2 3,
        4 5 6,
        7 8 9
    }
}

#[test]
fn test_matrix_macro() {
    let macro_matrix = matrix! {
        99   12     0,
        55   33485  30
    };
    assert_eq!(macro_matrix, Matrix::from([[99, 12, 0], [55, 33485, 30]]))
}

#[test]
fn test_matrix_display() {
    println!("{}", matrix_3x3_1to9())
}

#[test]
fn test_matrix_zero() {
    let zero_matrix = SquareMatrix::<3>::zero();
    assert_eq!(
        zero_matrix,
        matrix! {
            0 0 0,
            0 0 0,
            0 0 0
        }
    );
}

#[test]
fn test_matrix_identity() {
    let identity_matrix = SquareMatrix::<3>::identity();
    assert_eq!(
        identity_matrix,
        matrix! {
            1 0 0,
            0 1 0,
            0 0 1
        }
    );
}

#[test]
fn test_matrix_is_square() {
    let square_matrix = SquareMatrix::<5>::zero();
    assert!(square_matrix.is_square(), "Square matrix not square (how).")
}

#[test]
fn test_matrix_addition() {
    let m1 = matrix_3x3_1to9();
    let m2 = matrix_3x3_1to9();
    let m3 = m1 + m2;

    assert_eq!(
        m3,
        matrix! {
            2  4  6,
            8  10 12,
            14 16 18
        }
    );
}

#[test]
fn test_matrix_subtraction() {
    let m1 = matrix_3x3_1to9();
    let m2 = matrix_3x3_1to9();
    let m3 = m1 - m2;
    assert_eq!(m3, Matrix::zero());
}

#[test]
fn test_matrix_multiplication() {
    let m1_a = matrix! { 5 };
    let m1_b = matrix! { 3 };
    assert_eq!(
        m1_a * m1_b,
        Matrix::from([[15]]),
        "(1x1) * (1x1) matrix multiplication failed."
    );

    let m2_a = matrix! {
        1 2 3,
        4 5 6
    };
    let m2_b = matrix! {
        1 2,
        3 4,
        5 6
    };

    assert_eq!(
        m2_a * m2_b,
        matrix! {
            22 28,
            49 64
        },
        "(2x3) * (3x2) matrix multiplication failed."
    );
}

#[test]
fn test_matrix_determinant() {
    let m0 = Matrix::<0, 0>::zero();
    assert_eq!(m0.determinant(), 1.0, "(0x0) matrix determinant failed.");

    let m1 = matrix! { 123 };
    assert_eq!(m1.determinant(), 123.0, "(1x1) matrix determinant failed.");

    let m2 = matrix! {
        11 7,
        2  5
    };
    assert_eq!(m2.determinant(), 41.0, "(2x2) matrix determinant failed.");

    let m3 = matrix! {
        7 4  5,
        3 10 1,
        9 0  7
    };
    assert_eq!(m3.determinant(), -8.0, "(3x3) matrix determinant failed.");

    let m4 = matrix! {
        7 8  4 5,
        6 22 1 4,
        7 12 3 2,
        0 5  9 3
    };
    assert_eq!(m4.determinant(), 2921.0, "(4x4) matrix determinant failed.");
}

#[test]
fn test_matrix_inverse() {
    let matrix_where_determinant_is_0 = SquareMatrix::<3>::zero();
    assert!(!matrix_where_determinant_is_0.has_inverse());
}

#[test]
fn test_matrix_transpose() {
    let transposed_matrix = matrix! {
        1 2 3,
        4 5 6
    }
    .transpose();
    assert_eq!(
        transposed_matrix,
        matrix! {
            1 4,
            2 5,
            3 6
        }
    )
}

```