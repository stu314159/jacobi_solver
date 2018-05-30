#ifndef PYJACOBI_SOLVER_H_
#define PYJACOBI_SOLVER_H_

#include <boost/python.hpp>


class PyJacobi_Solver
{
public:
  PyJacobi_Solver(const int N);
  ~PyJacobi_Solver();
  void set_u_out(boost::python::object obj);
  void set_u_even(boost::python::object obj);
  void set_u_odd(boost::python::object obj);

  void set_maxIter(int mI);
  void set_tolerance(double tol);
  void set_rhs(double r);

  void solve();
  int get_iter();
  int get_exit_code();

  void print_status();


private:
  double * u_out;
  double * u_even;
  double * u_odd;

  double rel_error(const double * u, const double * u_new);

  const int N;
  int maxIter;
  double tolerance;
  int exit_code;
  int numIter;
  double rhs;
};

#endif
