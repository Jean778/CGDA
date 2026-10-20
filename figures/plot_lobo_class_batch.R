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
# LOBO (closed/base/budget=0.1) 类别-批次性能单图
# 6 类轨迹线 + 气泡点（大小=n）+ 总体黑线
# 低 F1 红圈、n=0 灰 ×、0.70 虚线
# 图例：n(气泡)在左、类别(点线)在右，水平并排顶端对齐，右下角无填充
# ============================================================
suppressMessages({library(ggplot2); library(dplyr); library(tidyr)})
cls_cols <- c(Ethanol="#A9D18E", Ethylene="#6FA9A6", Ammonia="#F4B183",
              Acetaldehyde="#D83C3C", Acetone="#7999B9", Toluene="#AFABAB")
theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.18, "cm"),
        axis.ticks         = element_line(linewidth = 0.8, color = "black"),
        axis.text          = element_text(face = "bold", size = 14, color = "black"),
        axis.title         = element_text(face = "bold", size = 14, color = "black"),
        legend.text        = element_text(face = "bold", size = 10),
        legend.title       = element_text(face = "bold", size = 11),
        legend.background  = element_blank(),
        panel.border       = element_blank(),
        panel.grid.major   = element_line(color = "grey90", linewidth = 0.4),   # 🟥 添加网格
        axis.line          = element_line(linewidth = 0.8, color = "black"))
m <- read.csv(file.path(REPO, "results/revision_v2/metrics_by_fold_seed.csv"), stringsAsFactors = FALSE)
d <- read.csv(file.path(REPO, "data/processed/drift_clean.csv"), stringsAsFactors = FALSE)
gmap <- d %>% distinct(gas_id, gas_name) %>% arrange(gas_id)
gas_lab <- gmap$gas_name
sub <- m %>% filter(family=="closed", setting=="lobo", method=="base", budget==0.1,
                    grepl("^f1_class", metric))
sub$cls <- as.integer(sub("f1_class_", "", sub$metric))
f1 <- sub %>% group_by(fold, cls) %>% summarise(F1 = mean(value), .groups="drop")
mc <- m %>% filter(family=="closed", setting=="lobo", method=="base", budget==0.1,
                   metric=="macro_f1") %>%
  group_by(fold) %>% summarise(M = mean(value), .groups="drop")
n_tab <- d %>% count(batch, gas_id, name="n") %>%
  tidyr::complete(batch = 1:10, gas_id = 1:6, fill = list(n = 0))
f1$batch <- f1$fold
f1$gas <- gas_lab[f1$cls]
f1 <- f1 %>% left_join(n_tab, by=c("batch"="batch","cls"="gas_id"))
f1$gas <- factor(f1$gas, levels = gas_lab)
f1$low_mark <- !is.na(f1$F1) & f1$F1 < 0.70
low <- f1 %>% filter(low_mark)
no_s <- f1 %>% filter(is.na(F1))

p <- ggplot(f1, aes(x = batch, y = F1)) +
  geom_hline(yintercept = 0.70, linetype = "dashed", color = "grey60", linewidth = 0.5) +
  geom_line(aes(color = gas), linewidth = 1.4, alpha = 0.85) +
  geom_point(aes(size = n, color = gas), shape = 21, fill = "white", stroke = 0.9) +
  geom_point(data = no_s, shape = 4, size = 3.2, stroke = 1.2, color = "grey45") +
  geom_point(data = low, aes(size = n), shape = 1, color = "#D83C3C", stroke = 1.3) +
  geom_line(data = mc, aes(x = fold, y = M), color = "black", linewidth = 1.6) +
  geom_point(data = mc, aes(x = fold, y = M), color = "black", size = 3, shape = 18) +
  scale_size_area(max_size = 8, name = "n") +
  scale_color_manual(values = cls_cols, name = NULL) +
  scale_x_continuous(breaks = 1:10, limits = c(0.8, 10.2)) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.25)) +
  labs(x = "Target Batch", y = "F1") +
  theme_pub +
  theme(legend.position = c(0.985, 0.025), legend.justification = c(1, 0),
        # 🟥 两个图例水平并排、顶端对齐
        legend.box = "horizontal", legend.box.just = "top",
        legend.direction = "vertical",
        legend.background = element_blank(),
        plot.margin = margin(4, 4, 4, 4)) +
  guides(size = guide_legend(order = 1,          # 🟥 气泡(n)图例在左
                             keywidth = unit(0.9, "cm"), keyheight = unit(0.55, "cm")),
         color = guide_legend(ncol = 2, order = 2,   # 🟥 点线(类别)图例在右
                              keywidth = unit(0.9, "cm"), keyheight = unit(0.85, "cm")))
p
# ggsave(file.path(REPO, "main_results/figures/lobo_class_batch.png"), p,
#        width = 8.4, height = 5.2, dpi = 300)