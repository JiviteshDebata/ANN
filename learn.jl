#Activation functions start:

function ReLuLeaky(result)		#Leaky Rectified Linear Unit implementation
	for i in(1:length(result))
		if(result[i])<0)
			result[i]=0.01*result[i];
		end
	end
end

function tanh(result)       #tanh implementation
  for i in (1:length(result))
    result[i]=((e^result[i]-(1/e^result[i]))/(e^result[i]+(1/e^result[i])))
  end
end

function sigmoid(result)    #sigmoid implementation
  for i in (1:length(result))
    result[i]=(1/(1+(e^result[i])))
  end
end

#Activation functions end

function compute(input,theta)
  a=[]
  push!(a,input)
  #println("input: $input")
  #println("theta1: $theta[1]")
  resulthere=theta[1] * input
  sigmoid(resulthere)
  push!(a,resulthere)
  for j in (2:length(theta))
    #println(j)
    resulthere=theta[j] * resulthere
    sigmoid(resulthere)
    push!(a,resulthere)
    #println("Resulthere: $resulthere")
  end
  return a
end

function backprop(a, output,theta)
  delta=[]; j=length(theta)
  #println("A[end]: $(size(a[end]))")
  #println("Output: $(size(output))")
  push!(delta,a[end]-output)
  for i in (length(a)-1:-1:1)
    #println("i $i")
    deltahere=((theta[j])'*delta[1]) .* (a[i] .* (1-a[i]))
    unshift!(delta,deltahere)
    j-=1
  end
  return delta
end
