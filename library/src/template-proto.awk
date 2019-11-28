# multi-line
BEGIN {
  RS = "\n\n+";
  FPAT = ".*^template.*ROCBLAS_EXPORT_NOINLINE.*_template(.*).*{.*"
}

{
  if (NF == 1) {
    gsub(/{.*/,"",$1); # strip out body
    gsub(/ROCBLAS_EXPORT_NOINLINE/,"",$1);
    gsub(/\)/,");",$1);
    print $1;
  }
}