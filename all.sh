#!/bin/bash

 

set -e

 

echo ""

echo "######## Running all reproducible code ############"

echo ""

 

echo ""

echo "######## Starting application on extreme Arabian PM2.5 concentrations (Section 5) ############"

echo ""

 

julia --project=. --threads=auto src/HW_cNBE.jl

 

echo ""

echo "######## Starting study using the inverted max-stable process (Section 4.3) ############"

echo ""

 

Rscript src/simulate_IMSP_random_tau.R

julia --project=. --threads=auto src/IMSP_random_tau_cNBE.jl

 

 

echo ""

echo "######## Everything finished! ############"

echo ""
