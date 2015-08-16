"""
  Author: Triskelion, HJ van Veen, info@mlwave.com

  Description:
  
    Here will be the experiments with compression of the bitstring.
	
	We compress the bitstring by chopping it up into n equally sized bit chunks.
	
	We then take the sum of the bit chunk, and look if that is over half the size of the bit chunk.
	 
	Example:
	
	original bitstring of size 10: 
	
	"1110101000"
	
	chop into 2 chunks of length 5.
	
	11101 = 4 = larger than 2.5 = 1
	01000 = 1 = smaller than 2.5 = 0
	
	output: 
	
	"10"
	
	Reference/Inspiration:
	
	Similarity Estimation Techniques From Rounding Algorithms, Moses Charikar
	http://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf
	
  Todo:
    
	Clean up code.
	Make much faster.
	Acquire probabilistic pseudo-random supercomputer
"""