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
# 4.8 部署可行性补充图：ECE（Dataset A / B / C 三面板）
# A=黄 B=绿 C=蓝标题；CGDA 青色柱、文献灰色柱；1 行 3 列
# ============================================================
suppressMessages({library(ggplot2); library(dplyr); library(patchwork)})
theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.18, "cm"),
        axis.ticks         = element_line(linewidth = 0.8, color = "black"),
        axis.text          = element_text(face = "bold", size = 11, color = "black"),
        axis.title         = element_text(face = "bold", size = 12, color = "black"),
        panel.border       = element_blank(),
        axis.line          = element_line(linewidth = 0.8, color = "black"))
# ---- A/C 数据 ----
d_ac <- read.csv(file.path(REPO, "main_results/table_46_deployment_light.csv"), stringsAsFactors = FALSE)
d_ac <- d_ac %>% filter(dataset %in% c("A", "C"))
# ---- B (CDSL) 数据：method=base 的文献 + method=cg_sup 的 CGDA，跨 direction 求平均 ----
d_b_raw <- read.csv(file.path(REPO, "main_results/table_cdsl_deployment_light.csv"), stringsAsFactors = FALSE)
d_b <- d_b_raw %>%
  filter(method %in% c("base", "cg_sup")) %>%
  filter(!is.na(ece)) %>%
  group_by(model) %>%
  summarise(ece = mean(ece), .groups = "drop") %>%
  mutate(dataset = "B", is_cgda = (model == "CGDA"))
# ---- 合并 ----
d_ac$is_cgda <- d_ac$method == "cg_sup"
d_ac <- d_ac %>% select(dataset, model, ece, is_cgda)
d_all <- rbind(d_ac, d_b)
mk <- function(dd, lab, title_col) {
  dd$model <- factor(dd$model, levels = dd$model[order(dd$ece)])
  ggplot(dd, aes(x = model, y = ece)) +
    geom_col(aes(fill = is_cgda), width = 0.68) +
    geom_text(aes(label = sprintf("%.3f", ece)),
              vjust = -0.5, size = 4.0, fontface = "bold") +
    scale_fill_manual(values = c("FALSE" = "grey78", "TRUE" = "#258C88"), guide = "none") +
    scale_y_continuous(limits = c(0, 0.62), breaks = seq(0, 0.6, 0.2), expand = c(0, 0)) +
    labs(title = lab, x = NULL, y = "ECE") +
    theme_pub +
    theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5, color = title_col),
          axis.text.x = element_text(face = "bold", size = 11, angle = 25, hjust = 0.8),
          plot.margin = margin(6, 10, 6, 6))
}
pA <- mk(d_all %>% filter(dataset == "A"), "Dataset A", "#ED7D31")   # 黄
pB <- mk(d_all %>% filter(dataset == "B"), "Dataset B", "#70AD47")   # 绿
pC <- mk(d_all %>% filter(dataset == "C"), "Dataset C", "#4472C4")   # 蓝
fig <- pA | pB | pC
fig
# ggsave(file.path(REPO, "main_results/figures/ece_46.png"), fig,
#        width = 14, height = 5, dpi = 300)
