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
# GSA-LC 数据集（浓度迁移）
# 正文主图：b（核心对比）+ c（配对变化）+ e（效率）
# 样式：每子图均有左 y + 下 x 轴，子图标题无框，图例在图下端
# ============================================================
suppressMessages({library(ggplot2); library(dplyr); library(tidyr); library(patchwork)})
if (!requireNamespace("ggh4x", quietly = TRUE)) install.packages("ggh4x")
suppressMessages(library(ggh4x))

theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.18, "cm"),
        axis.ticks         = element_line(linewidth = 0.8, color = "black"),
        axis.text          = element_text(face = "bold", size = 11, color = "black"),
        axis.title         = element_text(face = "bold", size = 13, color = "black"),
        strip.text         = element_text(face = "bold", size = 12),
        strip.background   = element_blank(),          # 🟥 子图标题无框
        panel.border       = element_blank(),          # 🟥 无四边框
        axis.line          = element_line(linewidth = 0.8, color = "black"),
        legend.text        = element_text(face = "bold", size = 10),
        legend.title       = element_blank(),
        legend.background  = element_blank())

method_cols <- c(base = "#D83C3C", cal_sup = "#F4B183", mixup_std = "#CA704C",
                 cg_shuffle = "#7999B9", cg_sup = "#258C88")
method_labels <- c(base = "Source Only", cal_sup = "Cal-Sup", mixup_std = "Source+Mixup",
                   cg_shuffle = "CG-Shuffle", cg_sup = "CGDA")
method_order <- c("base", "cal_sup", "mixup_std", "cg_shuffle", "cg_sup")
conc_labels <- c("50" = "Target: 50 ppb", "100" = "Target: 100 ppb", "200" = "Target: 200 ppb")

d <- read.csv(file.path(REPO, "main_results/panel_gsa_loco_long.csv"))
s <- read.csv(file.path(REPO, "main_results/panel_gsa_loco_summary.csv"))
s$method <- factor(s$method, levels = method_order)
d$method <- factor(d$method, levels = method_order)

# ============ 正文主图 b：12 样本低预算 5 方法对比 ============
p_b <- ggplot(s, aes(x = method, y = macro_f1_mean * 100)) +
  geom_col(aes(color = method), fill = NA, width = 0.5, linewidth = 1.1) +
  geom_errorbar(aes(ymin = (macro_f1_mean - macro_f1_std) * 100,
                    ymax = (macro_f1_mean + macro_f1_std) * 100),
                width = 0.25, linewidth = 0.9) +
  geom_jitter(data = d, aes(x = method, y = macro_f1 * 100, color = method),
              width = 0.12, size = 1.8, alpha = 0.65) +
  scale_color_manual(values = method_cols, labels = method_labels) +
  ggh4x::facet_wrap2(~ target_conc, nrow = 1, axes = "all",   # 🟥 每子图均有 y 轴
                     labeller = labeller(target_conc = conc_labels)) +
  labs(x = NULL, y = "Macro-F1 (%)", subtitle = "Budget = 0.02 (n_cal = 12)") +
  theme_pub +
  theme(legend.position = "none",
        axis.text.x = element_text(face = "bold", size = 9, angle = 25, hjust = 1))

# ============ 正文主图 c：3 个浓度迁移方向配对变化 ============
pair <- d %>% filter(method %in% c("base", "cg_sup")) %>%
  mutate(dir_label = factor(target_conc, levels = c(50, 100, 200),
                            labels = c("50 <- 100+200", "100 <- 50+200", "200 <- 50+100")))
pair$method <- factor(pair$method, levels = c("base", "cg_sup"),
                      labels = c("Source Only", "CGDA"))
p_c <- ggplot(pair, aes(x = method, y = macro_f1 * 100, group = seed)) +
  geom_line(color = "grey70", linewidth = 0.45, alpha = 0.6) +
  geom_point(aes(color = method), size = 2.2) +
  scale_color_manual(values = c("Source Only" = "#D83C3C", "CGDA" = "#258C88")) +
  ggh4x::facet_wrap2(~ dir_label, nrow = 1, axes = "all") +
  labs(x = NULL, y = "Macro-F1 (%)") +
  theme_pub +
  theme(legend.position = "bottom",                      # 🟥 图例放图下端
        axis.text.x = element_text(face = "bold", size = 9)) +
  guides(color = guide_legend(nrow = 1))

# ============ 正文主图 e：性能 vs 推理延迟 ============
p_e <- ggplot(d, aes(x = seconds, y = macro_f1 * 100, color = method)) +
  geom_point(size = 2.6, alpha = 0.75) +
  scale_color_manual(values = method_cols, labels = method_labels) +
  ggh4x::facet_wrap2(~ target_conc, nrow = 1, axes = "all",
                     labeller = labeller(target_conc = conc_labels)) +
  labs(x = "Inference Time (s)", y = "Macro-F1 (%)") +
  theme_pub +
  theme(legend.position = "bottom") +                    # 🟥 图例放图下端
  guides(color = guide_legend(nrow = 1))

# ============ 组合 ============
p_b 
p_c 
p_e
# main <- (p_b / p_c / p_e) + plot_layout(heights = c(1, 1, 1))
# main
# ggsave(file.path(REPO, "main_results/figures/gsa_loco_main.png"), main,
#        width = 10, height = 13.5, dpi = 300)
