import numpy
import statistics as st
from scipy import stats
mu, sigma = 0, 1
samples1 = numpy.random.normal(mu, sigma, 1000)
X = numpy.array_split(samples1, 10)
print("The below stats is for {0} sets and {1} samples each".format(10,100))
for num,arr in enumerate(X, start=1):
    mean = st.mean(arr)
    stdev = st.stdev(arr)
    mle = stats.norm(mean, stdev).pdf(mean)
    print("The maximum likelihood of sample {0} and mean {1} and standard deviation {2} is {3}".format(num,mean,stdev,mle))

print('\n')
samples2 = numpy.random.normal(mu, sigma, 2000)
X = numpy.array_split(samples2, 10)
print("The below stats is for {0} sets and {1} samples each".format(10,200))
for num,arr in enumerate(X, start=1):
    mean = st.mean(arr)
    stdev = st.stdev(arr)
    mle = stats.norm(mean, stdev).pdf(mean)
    print("The maximum likelihood of sample {0} and mean {1} and standard deviation {2} is {3}".format(num,mean,stdev,mle))

#print('\n')
#samples2 = numpy.random.normal(mu, sigma, 500)
#X = numpy.array_split(samples2, 5)
#print("The below stats is for {0} sets and {1} samples each".format(5,100))
#for num,arr in enumerate(X, start=1):
#    mean = st.mean(arr)
#    stdev = st.stdev(arr)
#    mle = stats.norm(mean, stdev).pdf(mean)
#    print("The maximum likelihood of sample {0} and mean {1} and standard deviation {2} is {3}".format(num,mean,stdev,mle))
