#include "PyJacobi_Solver.h"
#include <math.h>
#include <iostream>
#include "WorkArounds.h"

PyJacobi_Solver::PyJacobi_Solver(const int N) :
u_out(NULL), u_even(NULL), u_odd(NULL), N(N), maxIter(0),
tolerance(0),exit_code(0),numIter(0),rhs(0)
{

}

PyJacobi_Solver::~PyJacobi_Solver()
{

}

void PyJacobi_Solver::set_u_out(boost::python::object obj)
{
  PyObject* pobj = obj.ptr();
  Py_buffer pybuf;
  PyObject_GetBuffer(pobj,&pybuf,PyBUF_SIMPLE);
  void * buf = pybuf.buf;
  u_out = (double*) buf;
}

void PyJacobi_Solver::set_u_even(boost::python::object obj)
{
  PyObject* pobj = obj.ptr();
  Py_buffer pybuf;
  PyObject_GetBuffer(pobj,&pybuf,PyBUF_SIMPLE);
  void * buf = pybuf.buf;
  u_even = (double*) buf;
}

void PyJacobi_Solver::set_u_odd(boost::python::object obj)
{
  PyObject* pobj = obj.ptr();
  Py_buffer pybuf;
  PyObject_GetBuffer(pobj,&pybuf,PyBUF_SIMPLE);
  void * buf = pybuf.buf;
  u_odd = (double*) buf;
}

void PyJacobi_Solver::set_maxIter(int maxI)
{
  maxIter = maxI;
}

void PyJacobi_Solver::set_tolerance(double tol)
{
  tolerance = tol;
}

void PyJacobi_Solver::set_rhs(double r)
{
  rhs = r;
}

double PyJacobi_Solver::rel_error(const double * u, const double * u_new)
{
  double norm_err = 0.; //initialize the error norm
  double norm_u = 0.; //initialize the vector norm
#pragma acc data update host(u_odd,u_even) 
 for(int i = 0; i<N; i++){
    norm_err+=(u[i] - u_new[i])*(u[i] - u_new[i]);
    norm_u+=u_new[i]*u_new[i];
  }
  double rel_err = sqrt(norm_err)/sqrt(norm_u);
  return rel_err;
}

void PyJacobi_Solver::solve()
{
  int nIter = 0;
  exit_code = 0;
  bool KEEP_GOING = true;
  double * u; double * u_new;
  double rel_update = 1.;

  double * u_even = this->u_even;
  double * u_odd = this->u_odd;
  int N = this->N;

  dummyUse(N);
#pragma acc data copyin(u_even[0:N],u_odd[0:N])
  while(KEEP_GOING)
  {
    nIter++; //increment iteration counter
    if(nIter%2 == 0) // set pointers
    { 
      u = u_even; u_new = u_odd;
    }else{
      u = u_odd; u_new = u_even;
    }
#pragma acc kernels
{ 
    for(int i = 1; i<(N-1); i++){ //iterate through all points
      u_new[i] = 0.5*(u[i-1]+u[i+1] - rhs);
    }
} 
   if (nIter > 2)
    {
  //    rel_update = rel_error(u,u_new);
  //   refactor to put calculation in this function
  //
      double normUpdate = 0.;
      for(int i=1; i<(N-1);i++){
        normUpdate+=(u[i] - u_new[i])*(u[i] - u_new[i]);
      }

      double normU = 0.;
      for(int i=1; i<(N-1);i++){

        normU+=u_new[i]*u_new[i];
      }

      rel_update = normUpdate/normU;
    }
    if ((rel_update < tolerance) || (nIter == maxIter)) {
      KEEP_GOING = false;
      numIter = nIter;
      //before leaving, copy the solution to u
      for(int i = 0; i<N;i++){
        u_out[i] = u_new[i];
      }
      if (nIter == maxIter) {
        exit_code = -1;
      }

    } // check if I should stop


  }// while loop

}// solve()

int PyJacobi_Solver::get_iter()
{
  return numIter;
}

int PyJacobi_Solver::get_exit_code()
{
  return exit_code;
}

void PyJacobi_Solver::print_status()
{
  std::cout << "Status: " << std::endl;
  std::cout << "maxIter = " << maxIter << std::endl;
  std::cout << "tolerance = " << tolerance << std::endl;
  std::cout << "rhs = " << rhs << std::endl;
}

using namespace boost::python;
BOOST_PYTHON_MODULE(Jacobi_Solver)
{
  class_<PyJacobi_Solver>("PyJacobi_Solver",init<int>())
        .def("set_u_out",&PyJacobi_Solver::set_u_out)
        .def("set_u_even",&PyJacobi_Solver::set_u_even)
        .def("set_u_odd",&PyJacobi_Solver::set_u_odd)
        .def("set_maxIter",&PyJacobi_Solver::set_maxIter)
        .def("set_tolerance",&PyJacobi_Solver::set_tolerance)
        .def("solve",&PyJacobi_Solver::solve)
        .def("get_iter",&PyJacobi_Solver::get_iter)
        .def("get_exit_code",&PyJacobi_Solver::get_exit_code)
        .def("set_rhs",&PyJacobi_Solver::set_rhs)
        .def("print_status",&PyJacobi_Solver::print_status)
        ;
}
