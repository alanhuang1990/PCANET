#include <algorithm>
#include <cstdio>
#include <cmath>
#include "mex.h"

template<class T>
T max(T a, T b)
{
    return a>b? a:b;
}

class Mat2D {
private:
    double * p_data;
    int N_row,N_col;
    
public:
    Mat2D()
    {
        p_data = NULL;
        N_row = 0;
        N_col = 0;
    }
    Mat2D(double * ptr,int _n_row,int _n_col)
    {
        p_data = ptr;
        N_row = _n_row;
        N_col = _n_col;
    }
    int get_index(int i_row, int i_col)
    {
        if(i_row >= N_row || i_col >= N_col)
        {
            printf("try to access (%d %d)\n",i_row,i_col);
            mexErrMsgTxt("invalid index for Mat class in get_idx() ");
        }
        return i_col*N_row + i_row;
    }
    
    int get_N_col()
    {
        return this->N_col;
    }
    int get_N_row()
    {
        return this->N_row;
    }
    double & operator () (const int i_row, const int i_col)
    {
        return (this->p_data[get_index(i_row,i_col)]);
    }
    double & operator () (const int id)
    {
        if(id >= N_row*N_col)
        {
            mexErrMsgTxt("invalid index for Mat class in binary operator () ");
        }
        return this->p_data[id];
    }
    void init_zeros()
    {
        for (int j_col = 0; j_col < N_col; j_col ++) {
            for (int i_row = 0; i_row < N_row; i_row ++) {
                p_data[j_col*N_row + i_row] = 0;
            }
        }
    }
};


int half_row_width = 0;
int half_col_width = 0;
int row_connectivity = 0;
int col_connectivity = 0;


void mark_connectivity(Mat2D & mat_res, int row_id_a,int col_id_a, int row_id_b, int col_id_b, double value)
{
    //printf("(%d %d) is connected with (%d %d)\n",row_id_a,col_id_a,row_id_b,col_id_b);
    int id_a = col_id_a * row_connectivity + row_id_a;
    int id_b = col_id_b * row_connectivity + row_id_b;
    //printf("id_a: %d    id_b: %d \n",id_a,id_b);
    mat_res(id_a,id_b) = max(mat_res(id_a,id_b),value);
    mat_res(id_b,id_a) = mat_res(id_a,id_b);
}

void  make_connectivity_at( Mat2D &filter, Mat2D & mat_res,int row_id, int col_id)
{
    for (int j_col =  col_id - half_col_width ; j_col <= col_id+half_col_width; j_col++) {
        for (int i_row = row_id - half_row_width; i_row <= row_id+half_row_width; i_row++) {
            if (i_row < 0 || i_row >= row_connectivity || j_col < 0 || j_col >= col_connectivity) {
                continue;
            }
            /*
            int id_a = col_id * row_connectivity + row_id;
            int id_b = j_col * row_connectivity + i_row;
            //printf("id_a: %d    id_b: %d \n",id_a,id_b);
            mat_res(id_a,id_b) = max(mat_res(id_a,id_b),value);
            mat_res(id_b,id_a) = mat_res(id_a,id_b);
            */
            mark_connectivity(mat_res,row_id,col_id,i_row,j_col,filter(i_row - row_id + half_row_width, j_col - col_id + half_col_width ));
        }
    }
}

void  make_connectivity(Mat2D &filter, Mat2D & mat_res)
{
    for (int j_col =0; j_col <col_connectivity; j_col++) {
        for (int i_row=0; i_row < row_connectivity; i_row++) {
            make_connectivity_at(filter, mat_res,i_row , j_col);
        }
    }
}

// input:   1. matrix from 'im = im2col_general'
//          2. patch size [ a,b ]
//          3. image size [a, b]
//          4. connectivity filter F with size of M*N, M and M both are odd
// output:  1. the connectivity matrix


void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
    double * p_size = mxGetPr(prhs[1]);
    double * img_size = mxGetPr(prhs[2]);
    Mat2D filter = Mat2D(mxGetPr(prhs[3]),mxGetM(prhs[3]),mxGetN(prhs[3]));
    Mat2D mtx_in= Mat2D(mxGetPr(prhs[0]),mxGetM(prhs[0]),mxGetN(prhs[0]));
    
    plhs[0] = mxCreateDoubleMatrix(mtx_in.get_N_col(),mtx_in.get_N_col(),mxREAL);
    Mat2D mtx_out= Mat2D(mxGetPr(plhs[0]),mxGetM(plhs[0]),mxGetN(plhs[0]));
    mtx_out.init_zeros();
    
    half_row_width = filter.get_N_row()/2;
    half_col_width = filter.get_N_col()/2;
    row_connectivity = img_size[0]-p_size[0]+1;
    col_connectivity = img_size[1] - p_size[1] +1;
    
    //printf("row_connectivity: %d  col_connectivity:%d \n ",row_connectivity,col_connectivity);
    
    if (row_connectivity * col_connectivity != mtx_out.get_N_col()) {
        printf("warning: row_connectivity * col_connectivity = %d while  mtx_out.get_N_col() = %d \n",row_connectivity * col_connectivity ,mtx_out.get_N_col());
    }
    
    
    
    make_connectivity(filter,mtx_out);
    
    
    
}
