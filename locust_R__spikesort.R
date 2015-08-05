
# Load the data
dN = paste("Locust_", 1:4,".dat.gz",sep="")
nb = 20*15000
readLocustFile <- function(file,nb){
   mC = gzfile(file,open="rb")
   x = readBin(mC,what="double",n=nb)
   close(mC)
   return (x)
}

# lD = list()
# for (i in 1:length(dN)){
#    lD[i] = readLocustFile(dN[i], nb)
# }


lD <- sapply(dN, function(n){
   mC = gzfile(n, open="rb")
   x = readBin(mC, what="double", n=nb)
   close(mC)
   return(x)
})

# Give names to each column:
colnames(lD) = paste("site",1:4)


# Check dimensions of data:
dim(lD)


# source utility functions
source("sorting_with_r.R")


# 5 number summmary
summary(lD, digits=2)


# Data were normalized:
apply(lD,2,sd)



# Digitization size:
apply(lD, 2, function(x) min(diff(sort(unique(x)))))


# How much saturation? Check # of consecutive samples that have 
# the same values
ndL = lapply(1:4, function(i) cstValueSgts(lD[,i]))
sapply(ndL, function(l) max(l[2]))


# Redefine matrix as a time series object
lD = ts(lD, start=0, freq=15e3)


# Plot the data:
plot(lD)


# Look at data on finer scale:
plot(window(lD,start=0,end=0.2))


# Renormalized based on median absolute deviation
lD.mad = apply(lD,2,mad)
lD = t((t(lD)-apply(lD,2,median))/lD.mad)
lD = ts(lD, start=0,freq=15e3)


# Plot the data and compare MAD to SD threshold:
plot(window(lD[,1],0,0.2))
abline(h=c(-1,1),col=2)
abline(h=c(-1,1)*sd(lD[,1]),col=4,lty=2,lwd=2)



# Detect spikes. Apply smoothing filter, keep parts above
# threshold

lDf = filter(lD, rep(1,5)/5)
lDf[is.na(lDf)] = 0
lDf.mad = apply(lDf,2,mad)
lDf = t(t(lDf)/lDf.mad)
thr = rep(4,4)
below.thr = t(t(lDf) < thr)
lDfr = lDf
lDfr[below.thr] = 0
remove(lDf)


# Plot the deteceted spikes over a short segment. Examine
# the first column of data.
plot(window(lD[,1],0,0.2))
abline(h=4,col=4,lty=2,lwd=2)
lines(window(ts(lDfr[,1],start=0,freq=15e3),0,0.2),col=2)


# Spikes are detected as local maxima on summed, filtered,
# and rectified trace
sp0 = peaks(apply(lDfr,1,sum),15)


# Split data into two parts:
(sp0E = as.eventsPos(sp0[sp0 <= dim(lD)[1]/2]))
(sp0L = as.eventsPos(sp0[sp0 > dim(lD)[1]/2]))


# Determine cut size for spike sorting. This means finding how long
# is the action potential shape. We need to determine the number of
# points before and after the peak that should be used to classify
# spike shapes

evtsE = mkEvents(sp0E, lD, 49, 50)
evtsE.med = median(evtsE)
evtsE.mad = apply(evtsE,1,mad)

plot(evtsE.med, type="n",ylab="Amplitude")
abline(v=seq(0,400,10),col="grey")
abline(h=c(0,1),col="grey")
lines(evtsE.med,lwd=2)
lines(evtsE.mad,col=2,lwd=2)



# Get Events with correct time spans
evtsE = mkEvents(sp0E,lD,14,30)

summary(evtsE)


# Show events using printing method
evtsE[,1:200]


# Assess noise by getting samples that are far away
# from spike events
noiseE = mkNoise(sp0E,lD,14,30,safetyFactor = 2.5,2000)

# Examine noise signatures
summary(noiseE)



# Function to extract isolated spike events
goodEvtsFct <- function(samp,thr=3) {
   samp.med <- apply(samp,1,median)
   samp.mad <- apply(samp,1,mad)
   above <- samp.med > 0
   samp.r <- apply(samp,2,function(x) {x[above] <- 0;x})
   apply(samp.r,2,function(x) all(abs(x-samp.med) < thr*samp.mad))
}


# Get isolated spike events
goodEvts = goodEvtsFct(evtsE,8)


# Plot a sample of the events
evtsE[,goodEvts][,1:200]



# Data live in 180 dimensional space: 45 sampling points
# and 4 recording sites

# Reduce data using Principal Components Analysis
evtsE.pc = prcomp(t(evtsE[,goodEvts]))


# Explore PCA
layout(matrix(1:4,nr=2))
explore(evtsE.pc,1,5)
explore(evtsE.pc,2,5)
explore(evtsE.pc,3,5)
explore(evtsE.pc,4,5)



# Static representation of the projected data
panel.dens = function(x, ...){
   usr = par("usr")
   on.exit(par(usr))
   par(usr=c(usr[1:2],0,1.5))
   d = density(x, adjust=0.5)
   x = d$x
   y = d$y
   y = y / max(y)
   lines(x,y,col="grey50")
}


pairs(evtsE.pc$x[,1:4],pch=".",gap=0,diag.panel=panel.dens)




# Clustering using kmeans. This is one part the could
# be improved. There are better ways to determine the number
# of clusters than visual inspection.
set.seed(1, kind="Mersenne-Twister")
km10 = kmeans(evtsE.pc$x[,1:3],centers=10,iter.max=100,nstart=100)
c10 = km10$cluster


# Order the clusters by size
cluster.med = sapply(1:10,function(cIdx) median(evtsE[,goodEvts][,c10==cIdx]))
sizeC = sapply(1:10, function(cIdx) sum(abs(cluster.med[,cIdx])))
newOrder = sort.int(sizeC, decreasing=TRUE,index.return=TRUE)$ix
cluster.mad = sapply(1:10,function(cIdx) {ce = t(evtsE)[goodEvts,];ce=ce[c10==cIdx,];apply(ce,2,mad)})
cluster.med = cluster.med[,newOrder]
cluster.mad = cluster.mad[,newOrder]
c10b = sapply(1:10, function(idx) (1:10)[newOrder=idx])[c10]



# Cluster specific plots
layout(matrix(1:4,nr=4))
par(mar=c(1,1,1,1))
plot(evtsE[,goodEvts][,c10b==1],y.bar=5)
plot(evtsE[,goodEvts][,c10b==2],y.bar=5)
plot(evtsE[,goodEvts][,c10b==3],y.bar=5)
plot(evtsE[,goodEvts][,c10b==4],y.bar=5)



# We stop here. This is a good run through of how neural data is analyzed
# to extract spike shapes, where each spike shape, or cluster, corresponds
# to  a single neuron.

# To finish the analysis, the spike times need to be extracted. The
# procedure of Pouzat has some manual heuristic aspects to the at the
# of it, which we'll skip.