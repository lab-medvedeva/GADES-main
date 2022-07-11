dyn.load("mtrx.so")
source("mtrx.R")


library(amap)
library(edgeR)
library(biomaRt)
library(Hobotnica)
library(MASS)
library('Matrix')
library(glue)
args = commandArgs(trailingOnly=TRUE)

countMatrixFile <- args[1]
annotationFile <- args[2]
mode <- args[3]
signature_file <- args[4]
save_to <- args[5]
save_metrics <- args[6]
num_samples <- strtoi(args[7])

signature <- read.csv(signature_file)$gene

print(signature)
print("SIGNATURE LENGTH")

print(length(signature))
if (mode == "ensembl") {
mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
signature_ensembl <- getBM(values=signature,
			    filters = "external_gene_name",
			    mart = mart,
  attributes = c("external_gene_name", "entrezgene_id",
    "hgnc_symbol", "description",
    "chromosome_name", "strand", "ensembl_gene_id"))$ensembl_gene_id

print("ENSEMBL PAM50 Signature")
} else if (mode != "gid") {
	stop ("mode should be one of: `ensembl`, `gid`. Exiting")
}

anno_raw = read.table(annotationFile, header=TRUE, sep=",")

anno = data.frame(anno_raw$cluster, row.names = anno_raw$cell)
#anno = data.frame(anno_raw$tissue, row.names = anno_raw$cell_id)

colnames(anno) <- "group"
print(rownames(anno)[1:10])
print(head(anno))

print('Reading table')

cm <- read.table(countMatrixFile, header=T, sep=",")
cm <- data.frame(cm[, -1], row.names=cm[, 1])
print("COUNT MATRIX SHAPE")
print(dim(cm))

print(rownames(anno)[1:10])
print('CM')
print(colnames(cm)[1:10])

#rownames(anno) <- gsub('-', '.', rownames(anno))
cat("SHARED SAMPLES SIZE: ")
cat(length(intersect(colnames(cm), rownames(anno))))
cat("\n")

#cm <- cm[, intersect(colnames(cm), rownames(anno))]
#print(cm)
#cm_cpm <- cpm(cm)
cm_cpm <- cpm(cm)
randomSignaturesList <- list()
randomSigScores <- list()
print('Ensembl')
#print(signature)
#print(rownames(cm_cpm)[1:100])

if (mode == "ensembl") {
	set_filt <- intersect(rownames(cm_cpm), signature_ensembl)
} else  if (mode == "gid"){
	set_filt <- intersect(rownames(cm_cpm), signature)
} else {
	stop("FUBAR")
}
print('Filt')
print(set_filt, dim(set_filt))


cm_cpm_original <- cm_cpm[set_filt, ]
print("CM SUBSET SHAPE")
print(rownames(cm_cpm_original))
print(dim(cm_cpm_original))

#print(randomSigScores)

distMatrix <- as.matrix(mtrx_Kendall_distance(as.matrix(cm_cpm_original), batch_size=20000)) #as.matrix(Dist(t(cm_cpm), method="kendall", nbproc=10))
#distMatrix <- as.matrix(Dist(t(cm_cpm), method="euclidean", nbproc=24))
score <- Hobotnica(distMatrix, anno$group)
write.csv(as.data.frame(as.matrix(distMatrix)), file=paste0(countMatrixFile, ".pam50.distmatrix"))
cat("SCORE: ")
cat(score)
cat("\n")


randomScores <- rep(0, num_samples)
for (i in 1:num_samples) {
    randomSignaturesList[[paste0("random_", i)]] <- sample(rownames(cm_cpm), length(signature))
}

cnt <- 0
for (name in names(randomSignaturesList)) {
    set_filt <- randomSignaturesList[[name]]
    xxx <- as.matrix(cm_cpm[set_filt, ])
    distMatrix <- as.matrix(mtrx_Kendall_distance(xxx, batch_size = 20000))
    h_score = Hobotnica(distMatrix, anno$group)
                print(h_score)
    randomSigScores[[name]] <- h_score
    randomScores[cnt] <- randomSigScores[[name]]
    cnt = cnt + 1
}


write.csv(as.matrix(randomSigScores), save_metrics)
pval <- min(1, (length(which(unlist(randomSigScores) >= score)) + 1)/length(unlist(randomSigScores)))
cat("P-value: ")
cat(pval)
cat("\n")


print(pval)

pdf(file = save_to, width = 8, height = 8)

hist(randomScores[1:num_samples - 1], breaks=seq(0.3,1.1,l=40), main=glue('H-score for markers: p-val {pval}'), xlab='H-score')
abline(v=score, col="blue")

text(x = score + 0.1, y = num_samples / 4, label = glue("H-score: {round(score, 3)}"), srt = 90)

dev.off()
