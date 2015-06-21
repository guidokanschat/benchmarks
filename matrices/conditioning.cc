/**
 * @file Compute condition numbers of cell mass and stiffness matrices
 * for a bunch of finite element shape function sets.
 *
 * @note This file  is part of https://github.com/guidokanschat/benchmarks.git
 */

#include <iostream>
#include <fstream>

#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_bernstein.h>
#include <deal.II/lac/lapack_full_matrix.h>

using namespace dealii;

enum class MatrixTypes
{
    mass,
    stiffness
};

template <int dim>
void matrix (const FiniteElement<dim>& element,
	     const MatrixTypes type)
{
  Triangulation<dim> tr;
  GridGenerator::hyper_cube (tr, 0., 1.);
  DoFHandler<dim> dh (tr);
  dh.distribute_dofs (element);
  
  const unsigned int degree = element.tensor_degree();
  QGauss<dim> quadrature(degree+1);
  LAPACKFullMatrix<double> M(element.dofs_per_cell);

  FEValues<dim> fe(element, quadrature,
		   update_JxW_values | update_values | update_gradients);
  fe.reinit(dh.begin_active());

  for (unsigned int k=0;k<quadrature.size();++k)
    for (unsigned int i=0;i<M.m();++i)
      for (unsigned int j=0;j<M.n();++j)
	for (unsigned int d=0;d<element.n_components();++d)
	  {
	    switch (type)
	      {
	    	case MatrixTypes::mass:
		      M(i,j) += fe.JxW(k)
				* fe.shape_value_component(j,k,d)
				* fe.shape_value_component(i,k,d);
	      	      break;
	      	case MatrixTypes::stiffness:
	      	      M(i,j) += fe.JxW(k)
	      			* (fe.shape_grad_component(j,k,d)
	      			   * fe.shape_grad_component(i,k,d));
	      	      break;
	      }
	  }

  M.compute_eigenvalues();
  double lmin = 1.e30;
  double lmax = -1.;
  
  for (unsigned int i=0;i<M.m();++i)
    {  
      const double lambda = M.eigenvalue(i).real();
if (std::fabs(lambda) >= 1.e-9)
	{
	  if (lambda < lmin)
	    lmin = lambda;
	  if (lambda > lmax)
	    lmax = lambda;
	}
    }
  deallog << "\t[" << lmin << "\t, " << lmax
	  << "]\tcond " << lmax/lmin;
  deallog << std::endl;
}


template<int dim>
void doit ()
{
  typedef std::shared_ptr<const FiniteElement<dim> > FEPtr;
  std::vector<FEPtr> elements;
  elements.push_back(FEPtr(new FE_Q<dim>(1)));
  elements.push_back(FEPtr(new FE_Q<dim>(2)));
  elements.push_back(FEPtr(new FE_Q<dim>(3)));
  elements.push_back(FEPtr(new FE_Q<dim>(4)));
  elements.push_back(FEPtr(new FE_Q<dim>(5)));
  elements.push_back(FEPtr(new FE_Q<dim>(6)));
  elements.push_back(FEPtr(new FE_Q<dim>(7)));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGauss<1>(1))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGauss<1>(2))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGauss<1>(3))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGauss<1>(4))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGauss<1>(5))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGauss<1>(6))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGauss<1>(7))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGauss<1>(8))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(2))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(3))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(4))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(5))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(6))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(7))));
  elements.push_back(FEPtr(new FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(8))));
  elements.push_back(FEPtr(new FE_Bernstein<dim>(1)));
  elements.push_back(FEPtr(new FE_Bernstein<dim>(2)));
  elements.push_back(FEPtr(new FE_Bernstein<dim>(3)));
  elements.push_back(FEPtr(new FE_Bernstein<dim>(4)));
  elements.push_back(FEPtr(new FE_Bernstein<dim>(5)));
  elements.push_back(FEPtr(new FE_Bernstein<dim>(6)));
  elements.push_back(FEPtr(new FE_Bernstein<dim>(7)));

for (const FEPtr& fe : elements)
  {
deallog << fe->get_name() << "::mass ";
matrix(*fe, MatrixTypes::mass);
deallog << fe->get_name() << "::stiff";
matrix(*fe, MatrixTypes::stiffness);
}

}


int main ()
{
  doit<2> ();
deallog << std::endl;
  doit<3> ();
}

