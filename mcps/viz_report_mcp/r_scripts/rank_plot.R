#!/usr/bin/env Rscript
# Rank vs Combined Score (Avg.Z) plot.
# Follows the design of TOmicsVis::gene_rank_plot (see NOTICE.md), written inline with
# ggplot2 + ggrepel so that TOmicsVis is not needed; takes CLI args.

suppressPackageStartupMessages({
  library(optparse); library(ggplot2); library(ggrepel)
})

op <- OptionParser(option_list = list(
  make_option("--integrated", type="character",
              help="INTEGRATED_ranked.csv (must have Gene, Combined Score (Avg.Z), Rank_Combined_Score, Pcom)"),
  make_option("--output",     type="character", help="output PNG path"),
  make_option("--top_n",      type="integer", default=10,
              help="number of top up + top down genes to label"),
  make_option("--width",      type="numeric", default=8),
  make_option("--height",     type="numeric", default=8),
  make_option("--dpi",        type="integer", default=180),
  make_option("--label_size", type="numeric", default=3)
))
o <- parse_args(op)
stopifnot(!is.null(o$integrated), !is.null(o$output))

df <- read.csv(o$integrated, check.names = FALSE)
# Normalise the score column name (could be "Combined Score (Avg.Z)" or
# "Combined.Score..Avg.Z." depending on read.csv()'s make.names())
score_col <- intersect(c("Combined Score (Avg.Z)", "Combined.Score..Avg.Z."),
                      names(df))[1]
stopifnot(!is.na(score_col))

plot_input <- data.frame(
  gene     = df$Gene,
  log2FC   = df[[score_col]],
  pvalue   = df$Rank_Combined_Score,
  padj     = df$Pcom,
  stringsAsFactors = FALSE
)
plot_input <- plot_input[!duplicated(plot_input$gene), ]
rownames(plot_input) <- plot_input$gene
plot_input <- plot_input[order(-plot_input$log2FC, plot_input$pvalue), ]
plot_input$rank <- seq_len(nrow(plot_input))
plot_input$log2FC_abs <- abs(plot_input$log2FC)

plot_input$color <- "black"
plot_input$color[plot_input$rank <= 10] <- "red"
plot_input$color[plot_input$rank >= (nrow(plot_input) - 9)] <- "blue"
plot_input$color <- factor(plot_input$color, levels = c("red","black","blue"))

top_up   <- rownames(head(plot_input, o$top_n))
top_down <- rownames(tail(plot_input, o$top_n))

p <- ggplot(plot_input, aes(x = rank, y = log2FC,
                            color = color, size = log2FC_abs)) +
  geom_point() +
  scale_color_manual(values = c("red"="red", "black"="black", "blue"="blue")) +
  geom_hline(yintercept = c(-1, 1), linetype = 2, size = 0.3) +
  geom_hline(yintercept = 0, linetype = 1, size = 0.5) +
  geom_vline(xintercept = median(plot_input$rank), linetype = 2, size = 0.3) +
  geom_text_repel(data = plot_input[top_up, ],
                  aes(label = gene), size = o$label_size, color = "red",
                  max.overlaps = Inf, box.padding = 0.5,
                  point.padding = 0.3, segment.color = "red",
                  segment.size = 0.3, force = 2) +
  geom_text_repel(data = plot_input[top_down, ],
                  aes(label = gene), size = o$label_size, color = "blue",
                  max.overlaps = Inf, box.padding = 0.5,
                  point.padding = 0.3, segment.color = "blue",
                  segment.size = 0.3, force = 2) +
  xlab("Rank") + ylab("Combined Score (Avg.Z)") +
  guides(color = "none", size = "none") +
  ylim(c(-max(abs(plot_input$log2FC)), max(abs(plot_input$log2FC)))) +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5), panel.grid = element_blank())

dir.create(dirname(o$output), showWarnings = FALSE, recursive = TRUE)
ggsave(o$output, plot = p,
       width = o$width, height = o$height, dpi = o$dpi, bg = "white")
cat(sprintf("✓ wrote %s (n=%d)\n", o$output, nrow(plot_input)))
