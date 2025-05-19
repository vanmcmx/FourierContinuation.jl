export BoundaryCondition, DirichletBC, NeumannBC

abstract type BoundaryCondition{T} end

"""
    DirichletBC
"""
struct DirichletBC{T} <: BoundaryCondition{T} end


"""
    NeumannBC
"""
struct NeumannBC{T} <: BoundaryCondition{T} end