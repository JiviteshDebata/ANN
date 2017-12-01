include("./io.jl")
include("./learn.jl")

hidden=[]
print("Enter file path for training data input: ")
input=trainio(1)
print("Enter file path for training data output: ")
output=trainio(2)
hideio(hidden)
#println(hidden)
theta=[]
if length(hidden)>0
  push!(theta,rand(hidden[1],length(input[:,1])))
  #println("tlen: $(length(theta))")
else
  push!(theta,rand(length(output[:,1]),length(input)))
  #println("tlen: $(length(theta))")
end
for i in (1:length(hidden)-1)
  push!(theta,rand(hidden[i+1],hidden[i]))
  #println("tlen: $(length(theta))")
end
if length(hidden)>0
  push!(theta,rand(length(output[:,1]),hidden[end]))
  #println("tlen: $(length(theta))")
end
#println("Output length: $(length(output[:,1]))")
#println("Theta length: $((theta))")

print("How many iterations?: ")
n=parse(Int,readline(STDIN))
print("Rate?: ")
eta=parse(Float64,readline(STDIN))
print("Momentum rate?: ")
alpha=parse(Float64,readline(STDIN))
for i in (1:n)
  oldDelta=[]; Delta=[]
  for k in (1:length(hidden)+1)
    push!(Delta,[])
  end
  m=length(input[1,:])
  for j in (1:m)
    send=input[:,j]
    receive=output[:,j]
    a=compute(send,theta)
    delta=backprop(a,receive,theta)
    for l in (1:length(theta))
      here=(delta[l]*a[l+1]')
      try
        Delta[l]+=here'
      catch
        Delta[l]=here'
      end
    end
  end
  
  for k in (1:length(theta))
    try
      oldDelta[k]=Delta[k]
    catch
      push!(oldDelta,Delta[k])
    end
    try  #Add regularisation
      theta[k]-=(eta/m)*Delta[k] - (alpha*oldDelta[k])
    catch
      theta[k]-=(eta/m)*Delta[k]
    end
  end
end

println("How many times would you like to test it out: ")
n=parse(Int,readline(STDIN))
for j in (1:n)
  print("Enter input : ")
  input=split(chomp(readline(STDIN)))
  input=map(input) do x
    x=parse(Int,x)
  end
  push!(input,1)
  a=compute(input,theta)
  println("\nOutput: ")
  for i in a[end]
    if (i>0.5)
      print("1 ")
    else
      print("0 ")
    end
  end
  println("\n")
end
