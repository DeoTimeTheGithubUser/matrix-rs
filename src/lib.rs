#![feature(inherent_associated_types)]

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Matrix<const R: usize, const C: usize>([[i32; C]; R]);

impl<
    const R: usize,
    const C: usize
> Matrix<R, C> {

    type Row = [i32; C];
    type Column = [i32; R];

    fn new(closure: impl Fn(usize, usize) -> i32) -> Self {
        Matrix(
            (0..R).map(|row| {
                (0..C)
                    .map(|column| closure(row, column))
                    .collect::<Vec<_>>().try_into().unwrap()
            }).collect::<Vec<_>>().try_into().unwrap()
        )
    }

    fn zero() -> Self { Matrix::new(|_, _| 0) }

    fn rows(self) -> [Self::Row; R] { self.0 }
    fn columns(self) -> usize { C }
    const fn is_square(self) -> bool { R == C }

    fn determinant(self) -> i32 {
        todo!()
    }

    fn inverse(self) -> Self {
        todo!()
    }

    fn merge(self, other: Matrix<R, C>, transform: impl Fn(i32, i32) -> i32) -> Self {
        Matrix::new(|row, column| transform(self[row][column], other[row][column]))
    }
}


impl<
    const R: usize,
    const C: usize
> Default for Matrix<R, C> {
    fn default() -> Self {
        Matrix::zero()
    }
}


type SquareMatrix<const D: usize> = Matrix<D, D>;
impl<const D: usize> SquareMatrix<D> {
    fn identity() -> Self {
        Matrix::new(|row, column| if row == column { 1 } else { 0 })
    }
}


impl<
    const R: usize,
    const C: usize
> std::ops::Index<usize> for Matrix<R, C> {
    type Output = Self::Row;

    fn index(&self, row: usize) -> &Self::Output {
        &self.0[row]
    }
}


impl<
    const R: usize,
    const C: usize
> std::ops::Add for Matrix<R, C> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.merge(rhs, |a, b| a + b)
    }
}

impl<
    const R: usize,
    const C: usize
> std::ops::Sub for Matrix<R, C> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.merge(rhs, |a, b| a - b)
    }
}

impl<
    const R: usize,
    const C: usize,
    const C2: usize
> std::ops::Mul<Matrix<C, C2>> for Matrix<R, C> {
    type Output = Matrix<R, C2>;

    fn mul(self, rhs: Matrix<C, C2>) -> Self::Output {
        todo!()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_matrix() -> Matrix<3, 3> {
        Matrix([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
    }

    #[test]
    fn test_matrix_zero() {
        let matrix = SquareMatrix::<3>::zero();
        assert_eq!(matrix, Matrix([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]))
    }

    #[test]
    fn test_matrix_identity() {
        let matrix = SquareMatrix::<3>::identity();
        assert_eq!(matrix, Matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]))
    }

    #[test]
    fn test_matrix_is_square() {
        let matrix = SquareMatrix::<5>::zero();
        assert!(matrix.is_square())
    }

    #[test]
    fn test_matrix_addition() {
        let m1 = dummy_matrix();
        let m2 = dummy_matrix();
        let m3 = m1 + m2;
        assert_eq!(m3, Matrix([
            [2, 4, 6],
            [8, 10, 12],
            [14, 16, 18]
        ]));
    }

    #[test]
    fn test_matrix_subtraction() {
        let m1 = dummy_matrix();
        let m2 = dummy_matrix();
        let m3 = m1 - m2;
        assert_eq!(m3, Matrix::zero());
    }

}

