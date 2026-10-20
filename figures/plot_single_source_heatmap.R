# ============================================================
# Repository root (REPO) resolves automatically in Rscript and RStudio.
# Override with env var, e.g.  REPO=<repo-root>  Rscript figures/plot_x.R
# ============================================================
if (Sys.getenv("REPO") != "") {
  REPO <- Sys.getenv("REPO")
} else if (interactive()) {
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    REPO <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
  } else {
    REPO <- getwd()
  }
} else {
  .cgda_args <- commandArgs(trailingOnly = FALSE)
  .cgda_file <- sub("^--file=", "", .cgda_args[grepl("^--file=", .cgda_args)])
  REPO <- if (length(.cgda_file) > 0) dirname(dirname(.cgda_file)) else getwd()
}
# ============================================================
# 严格单项跨域（Single-Source）：CGDA 精度三角热图（右侧轴版）
# budget = 0.1 | 指标 = accuracy（5 seed 均值，百分制）
# y 轴（Source Batch）在右侧；无顶部/左边框；色条嵌入左上三角
# ============================================================
suppressMessages({library(ggplot2); library(dplyr); library(tidyr)})
theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.18, "cm"),
        axis.ticks         = element_line(linewidth = 0.8, color = "black"),
        axis.text          = element_text(face = "bold", size = 11, color = "black"),
        axis.title         = element_text(face = "bold", size = 13, color = "black"),
        panel.border       = element_blank(),                              # 🟥 去掉全框
        axis.line          = element_line(linewidth = 0.8, color = "black"),
        legend.text        = element_text(face = "bold", size = 10),
        legend.title       = element_text(face = "bold", size = 11),
        legend.background  = element_blank())                              # 🟥 色条无背景
d <- read.csv(file.path(REPO, "main_results/panel_single_source_long.csv"), stringsAsFactors = FALSE)
cg <- d %>% filter(method == "cg_sup", abs(budget - 0.1) < 1e-8) %>%
  group_by(source_batch, target_batch) %>%
  summarise(acc = 100 * mean(accuracy), .groups = "drop")
cg$src <- factor(cg$source_batch, levels = 1:9)
cg$tgt <- factor(cg$target_batch, levels = 2:10)
p <- ggplot(cg, aes(x = tgt, y = src, fill = acc)) +
  geom_tile(color = "white", linewidth = 1.0) +
  geom_text(aes(label = sprintf("%.1f", acc)),
            size = 3.4, fontface = "bold", color = "black") +
  scale_fill_gradient(low = "#F6E6C2", high = "#A9C5DB",
                      limits = c(75, 100), name = "Accuracy (%)") +
  scale_y_discrete(position = "right") +                                   # 🟥 y 轴刻度移到右侧
  labs(x = "Target Batch", y = "Source Batch") +
  theme_pub +
  theme(legend.position = c(0.11, 0.62),                                   # 🟥 色条嵌入左上三角
        legend.justification = c(0.5, 0.5),
        legend.direction = "vertical",
        legend.key.height = unit(1.1, "cm"),
        legend.key.width = unit(0.55, "cm"),
        plot.margin = margin(8, 12, 8, 8))
p
# ggsave(file.path(REPO, "main_results/figures/single_source_cgda_heatmap.png"), p,
#        width = 8, height = 6.5, dpi = 300)
