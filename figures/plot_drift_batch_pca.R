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
# 数据集 A + B：漂移轨迹双图对比（1 行 2 列，宽度 2:1）
# ============================================================
# 🟨 首次运行需安装：install.packages(c("ggh4x", "patchwork"))
suppressMessages({library(ggplot2); library(ggh4x); library(dplyr); library(patchwork)})

theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.18, "cm"),
        axis.ticks         = element_line(linewidth = 0.8),
        axis.text          = element_text(face = "bold", size = 11),
        axis.title         = element_text(face = "bold", size = 14),
        legend.text        = element_text(face = "bold"),
        legend.title       = element_blank(),
        strip.text         = element_text(face = "bold", size = 12),
        strip.background   = element_rect(fill = "grey98", color = "black", linewidth = 0.7),
        panel.border       = element_rect(fill = NA, color = "black", linewidth = 0.7),
        panel.spacing      = unit(0.15, "cm"))

# ==================== 图 1：数据集 A 10 批次分面 PCA ====================
dA <- read.csv(file.path(REPO, "data/processed/drift_clean.csv"), check.names = FALSE)
fcols <- grep("^s", names(dA), value = TRUE)
cls_cols_A <- c(Acetaldehyde = "#D83C3C", Acetone = "#7999B9", Ammonia = "#F4B183",
                Ethanol = "#A9D18E", Ethylene = "#6FA9A6", Toluene = "#AFABAB")
pca <- prcomp(dA[, fcols], center = TRUE, scale. = TRUE)
pct <- round(100 * summary(pca)$importance[2, 1:2], 1)
pdA <- data.frame(batch = factor(dA$batch, levels = 1:10), gas = dA$gas_name,
                  PC1 = pca$x[, 1], PC2 = pca$x[, 2])
p1 <- ggplot(pdA, aes(PC1, PC2, color = gas)) +
  geom_point(size = 0.9, alpha = 0.6) +
  scale_color_manual(values = cls_cols_A) +
  ggh4x::facet_wrap2(~ batch, nrow = 2, axes = "all", scales = "free_y",
                     labeller = labeller(batch = ~ paste0("Batch ", .))) +
  labs(x = paste0("PC1 (", pct[1], "%)"), y = paste0("PC2 (", pct[2], "%)")) +
  theme_pub +
  theme(legend.position = "top", legend.justification = "right") +
  guides(color = guide_legend(nrow = 1))

# ==================== 1 行 2 列，宽度占比 2:1 ====================
# 🟥 图1（5列分面）占 2 份宽，图2（2列分面）占 1 份宽
p1

# ggsave(file.path(REPO, "R_figure/AB_drift_comparison.png"),
#        p1 + p2 + plot_layout(widths = c(2, 1)), width = 18, height = 8, dpi = 300)
