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
# 三图合并（1 行 3 列，统一字体）
# 图1: 三协议性能对比（UCI Drift，仅左 y + 下 x 轴）
# 图2: LOBO 类别-批次性能单图（气泡轨迹）
# 图3: 严格单项跨域 CGDA 精度三角热图（右侧轴 + 左上色条）
# ============================================================
suppressMessages({library(ggplot2); library(dplyr); library(tidyr); library(patchwork)})
# 🟥 统一主题：axis.text 12 / axis.title 13 / legend 10-11
theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.18, "cm"),
        axis.ticks         = element_line(linewidth = 0.8, color = "black"),
        axis.text          = element_text(face = "bold", size = 12, color = "black"),
        axis.title         = element_text(face = "bold", size = 13, color = "black"),
        panel.border       = element_blank(),
        axis.line          = element_line(linewidth = 0.8, color = "black"),
        legend.text        = element_text(face = "bold", size = 10, color = "black"),
        legend.title       = element_text(face = "bold", size = 11, color = "black"),
        legend.background  = element_blank())

# ============ 图 1：三协议性能对比（仅左 y + 下 x）============
dat <- read.csv(file.path(REPO, "docs/figures/panel_a_data.csv"), stringsAsFactors = FALSE)
col_mixed <- "#D83C3C"; col_lobo <- "#A9D4CC"; col_fwd <- "#CBBBDC"
df_drift_plot <- subset(dat, dataset == "UCI_Drift" & protocol %in% c("LOBO", "Forward"))
mixed_drift   <- subset(dat, dataset == "UCI_Drift" & protocol == "MixedCV")
p1 <- ggplot() +
  geom_hline(yintercept = mixed_drift$mean, colour = col_mixed,
             linetype = "dashed", linewidth = 1) +
  annotate("ribbon", x = c(0.5, 10.5),
           ymin = mixed_drift$mean - mixed_drift$sd,
           ymax = mixed_drift$mean + mixed_drift$sd,
           fill = col_mixed, alpha = 0.15) +
  annotate("text", x = 10.3, y = mixed_drift$mean + 0.02,
           label = sprintf("Mixed CV = %.3f", mixed_drift$mean),
           colour = col_mixed, fontface = "bold", size = 3.5, hjust = 1) +
  geom_point(data = df_drift_plot,
             aes(x = fold, y = mean, colour = protocol, shape = protocol), size = 3) +
  geom_errorbar(data = df_drift_plot,
                aes(x = fold, y = mean, ymin = mean - sd, ymax = mean + sd, colour = protocol),
                width = 0.2, linewidth = 0.8) +
  geom_line(data = df_drift_plot,
            aes(x = fold, y = mean, colour = protocol, linetype = protocol), linewidth = 1) +
  scale_colour_manual(values = c("LOBO" = col_lobo, "Forward" = col_fwd)) +
  scale_linetype_manual(values = c("LOBO" = "dashed", "Forward" = "dashed")) +
  scale_shape_manual(values = c("LOBO" = 16, "Forward" = 15)) +
  scale_x_continuous(breaks = 1:10, labels = paste0("B", 1:10)) +
  scale_y_continuous(limits = c(0.7, 1.05), breaks = seq(0.7, 1.0, 0.1)) +
  labs(x = "Target Batch", y = "Accuracy (MLP)") +
  theme_pub +
  theme(legend.position = c(0.17, 0.9), legend.title = element_blank(),
        panel.grid.major = element_line(color = "grey90", linewidth = 0.4),
        panel.grid.minor = element_blank())

 
# ============ 图 2：LOBO 类别-批次性能单图 ============
cls_cols <- c(Ethanol="#A9D18E", Ethylene="#6FA9A6", Ammonia="#F4B183",
              Acetaldehyde="#D83C3C", Acetone="#7999B9", Toluene="#AFABAB")
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
f1$batch <- f1$fold; f1$gas <- gas_lab[f1$cls]
f1 <- f1 %>% left_join(n_tab, by=c("batch"="batch","cls"="gas_id"))
f1$gas <- factor(f1$gas, levels = gas_lab)
f1$low_mark <- !is.na(f1$F1) & f1$F1 < 0.70
low <- f1 %>% filter(low_mark); no_s <- f1 %>% filter(is.na(F1))
p2 <- ggplot(f1, aes(x = batch, y = F1)) +
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
  theme(panel.grid.major = element_line(color = "grey90", linewidth = 0.4),
        legend.position = c(0.985, 0.025), legend.justification = c(1, 0),
        legend.box = "horizontal", legend.box.just = "top",
        legend.direction = "vertical",
        plot.margin = margin(4, 4, 4, 4)) +
  guides(size = guide_legend(order = 1, keywidth = unit(0.9, "cm"), keyheight = unit(0.55, "cm")),
         color = guide_legend(ncol = 2, order = 2, keywidth = unit(0.9, "cm"), keyheight = unit(0.85, "cm")))

# ============ 图 3：严格单项跨域三角热图 ============
d3 <- read.csv(file.path(REPO, "main_results/panel_single_source_long.csv"), stringsAsFactors = FALSE)
cg <- d3 %>% filter(method == "cg_sup", abs(budget - 0.1) < 1e-8) %>%
  group_by(source_batch, target_batch) %>%
  summarise(acc = 100 * mean(accuracy), .groups = "drop")
cg$src <- factor(cg$source_batch, levels = 1:9)
cg$tgt <- factor(cg$target_batch, levels = 2:10)
p3 <- ggplot(cg, aes(x = tgt, y = src, fill = acc)) +
  geom_tile(color = "white", linewidth = 1.0) +
  geom_text(aes(label = sprintf("%.1f", acc)),
            size = 3.2, fontface = "bold", color = "black") +
  scale_fill_gradient(low = "#F6E6C2", high = "#A9C5DB",
                      limits = c(75, 100), name = "Accuracy (%)") +
  scale_y_discrete(position = "right") +
  labs(x = "Target Batch", y = "Source Batch") +
  theme_pub +
  theme(legend.position = c(0.11, 0.62), legend.justification = c(0.5, 0.5),
        legend.direction = "vertical",
        legend.key.height = unit(1.1, "cm"), legend.key.width = unit(0.55, "cm"),
        plot.margin = margin(8, 12, 8, 8))

# ============ 1 行 3 列组合 ============

combined <- (p1 + plot_spacer() + p2 + plot_spacer() + p3) +
  plot_layout(widths = c(10, 0.4, 10, 0.4, 10))

combined
# ggsave(file.path(REPO, "main_results/figures/fig41_row3.png"), combined,
#        width = 16.5, height = 5.8, dpi = 300)
