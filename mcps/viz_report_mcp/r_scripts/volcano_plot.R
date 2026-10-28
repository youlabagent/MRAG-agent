#!/usr/bin/env Rscript
# log2FC (RNA) vs Combined Score (Avg.Z) volcano.
# Mirrors the original Plot/volcano.R but takes CLI args.

suppressPackageStartupMessages({
  library(optparse); library(ggplot2); library(ggrepel)
})

op <- OptionParser(option_list = list(
  make_option("--integrated", type="character",
              help="INTEGRATED_ranked.csv path"),
  make_option("--output",     type="character", help="output PNG path"),
  make_option("--width",      type="numeric", default=9),
  make_option("--height",     type="numeric", default=8),
  make_option("--dpi",        type="integer", default=180),
  make_option("--top_n",      type="integer", default=10,
              help="number of top up + top down genes to label")
))
o <- parse_args(op)
stopifnot(!is.null(o$integrated), !is.null(o$output))

df <- read.csv(o$integrated, check.names = FALSE)
score_col <- intersect(c("Combined Score (Avg.Z)", "Combined.Score..Avg.Z."),
                      names(df))[1]
stopifnot(!is.na(score_col))

df$color <- "black"
df$color[df$Rank_Combined_Score <= o$top_n] <- "red"
df$color[df$Rank_Combined_Score >= (nrow(df) - (o$top_n - 1))] <- "blue"
df$color <- factor(df$color, levels = c("red","black","blue"))

top_up   <- df[df$Rank_Combined_Score <= o$top_n, ]
top_down <- df[df$Rank_Combined_Score >= (nrow(df) - (o$top_n - 1)), ]

p <- ggplot(df, aes(x = log2FC, y = .data[[score_col]], color = color)) +
  geom_point(size = 2) +
  scale_color_manual(values = c("red"="red", "black"="black", "blue"="blue")) +
  geom_text_repel(data = top_up, aes(label = Gene), color = "red", size = 3,
                  max.overlaps = Inf, box.padding = 0.5,
                  segment.color = "red", segment.size = 0.3) +
  geom_text_repel(data = top_down, aes(label = Gene), color = "blue", size = 3,
                  max.overlaps = Inf, box.padding = 0.5,
                  segment.color = "blue", segment.size = 0.3) +
  geom_hline(yintercept = 0, linetype = 2, size = 0.3) +
  geom_vline(xintercept = 0, linetype = 2, size = 0.3) +
  xlab("Log2 Fold Change (RNA-seq)") + ylab("Combined Score (Avg.Z)") +
  guides(color = "none") +
  theme_bw() + theme(panel.grid = element_blank())

dir.create(dirname(o$output), showWarnings = FALSE, recursive = TRUE)
ggsave(o$output, plot = p,
       width = o$width, height = o$height, dpi = o$dpi, bg = "white")
cat(sprintf("✓ wrote %s\n", o$output))
