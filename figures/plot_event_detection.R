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
# 4.5 跨环境事件检测与部署边界（数据集 D = Air Quality UCI，余弦距离）
# a' = 事件比例条形图 | b = ΔAP 配对 | c' = 决策面箭头 | d = 时间序列
# 修改：b/c/d 图例入图内；c 加 ΔSens/ΔFAR 数值；d x 轴 2 周间隔、年份只写一次
# ============================================================
suppressMessages({library(ggplot2); library(dplyr); library(tidyr); library(patchwork)})
# 在 library 加载后、绘图前加这一行
Sys.setlocale("LC_TIME", "English")   # Windows 下生效；若失败用 "C"

theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.18, "cm"),
        axis.ticks         = element_line(linewidth = 0.8, color = "black"),
        axis.text          = element_text(face = "bold", size = 11, color = "black"),
        axis.title         = element_text(face = "bold", size = 13, color = "black"),
        strip.text         = element_text(face = "bold", size = 12),
        strip.background   = element_blank(),
        panel.border       = element_blank(),
        axis.line          = element_line(linewidth = 0.8, color = "black"),
        legend.text        = element_text(face = "bold", size = 10),
        legend.title       = element_blank(),
        legend.background  = element_blank())

base <- file.path(REPO, "main_results")
dirs <- c("0->1","0->2","0->3","1->2","1->3","2->3")

# ============ a'：事件比例条形图 ============
d <- read.csv(file.path(base, "panel_45d_timeseries_cosine.csv")); d$dir <- paste0(d$A,"->",d$B)
a <- read.csv(file.path(base, "panel_45a_ap_heatmap_cosine.csv"))
pr <- d %>% group_by(dir) %>% summarise(ratio = 100*mean(y_true==1), .groups="drop")
pr$dir <- factor(pr$dir, levels = dirs); pr$ap <- a$cg_full[match(as.character(pr$dir), a$direction)]
p_a <- ggplot(pr, aes(x = dir, y = ratio)) +
  geom_col(fill = "#9FB3C8", width = 0.62) +
  geom_text(data = pr %>% filter(as.character(dir) != "2->3"),
            aes(label = sprintf("AP %.4f", ap)), vjust = -0.6, size = 3.4, fontface = "bold") +
  annotate("text", x = 6, y = 90, label = "Majority-class\n(82.8%) \u2192 AP saturated",
           size = 3.2, fontface = "bold", color = "#D83C3C", hjust = 1) +
  scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0, 0.12))) +
  labs(x = "Transfer Direction", y = "Event Ratio (%)") + theme_pub +
  theme(axis.text.x = element_text(face = "bold", size = 11))

# ============ b：ΔAP 配对（图例图内右上，0->1 均值标注） ============
b <- read.csv(file.path(base, "panel_45b_delta_ap_cosine.csv")); b$direction <- paste0(b$A,"->",b$B)
b <- b %>% filter(method %in% c("cal_sup","cg_full"))
b$method <- factor(b$method, levels = c("cal_sup","cg_full"), labels = c("Cal-Sup","CGDA"))
b$dir <- factor(b$direction, levels = dirs)
b_sum <- b %>% group_by(direction, method) %>% summarise(m = mean(delta_AP)*100, .groups="drop")
b_sum$dir <- factor(b_sum$direction, levels = dirs)
lab_b <- b_sum %>% filter(direction == "0->1") %>%
  mutate(ny = ifelse(method == "CGDA", 0.9, -0.9))
p_b <- ggplot(b, aes(x = dir, y = delta_AP*100, color = method)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey55", linewidth = 0.7) +
  geom_line(aes(group = interaction(seed, direction)), color = "grey78", linewidth = 0.4, alpha = 0.55) +
  geom_jitter(width = 0.10, size = 2.2, alpha = 0.8) +
  geom_point(data = b_sum, aes(x = dir, y = m, color = method), shape = 18, size = 4.5, inherit.aes = FALSE) +
  geom_text(data = lab_b, aes(x = dir, y = m, label = sprintf("%+.2f", m), color = method),
            nudge_y = lab_b$ny, size = 3.6, fontface = "bold", inherit.aes = FALSE) +
  scale_color_manual(values = c("Cal-Sup" = "#F4B183", "CGDA" = "#258C88")) +
  labs(x = "Transfer Direction", y = expression(paste(Delta, "AP vs MLP (pp)"))) + theme_pub +
  theme(axis.text.x = element_text(face = "bold", size = 11),
        legend.position = c(0.82, 0.82), legend.justification = c(0.5, 0.5))

# ============ c'：决策面箭头（图例图内右下，ΔSens/ΔFAR 标注） ============
c <- read.csv(file.path(base, "panel_45c_pr_points_cosine.csv"))
c01 <- c %>% filter(A == 0, B == 1, method %in% c("mlp","threshold_only","cal_sup","cg_full"))
c01$method <- factor(c01$method, levels = c("mlp","threshold_only","cal_sup","cg_full"),
                     labels = c("MLP","Threshold-Only","Cal-Sup","CGDA"))
c_mean <- c01 %>% group_by(method) %>% summarise(sens = mean(event_sensitivity), far = mean(false_alarm_rate), .groups="drop")
mlp <- c_mean %>% filter(method == "MLP")
arrows <- c_mean %>% filter(method != "MLP") %>%
  mutate(x0 = mlp$far, y0 = mlp$sens,
         dfar = far - x0, dsens = sens - y0)
meth_col <- c("MLP" = "#D83C3C", "Threshold-Only" = "#7999B9", "Cal-Sup" = "#F4B183", "CGDA" = "#258C88")
p_c <- ggplot(c01, aes(x = false_alarm_rate, y = event_sensitivity, color = method)) +
  annotate("point", x = 0, y = 1, shape = 4, size = 4.5, color = "grey40", stroke = 1.2) +
  annotate("text", x = 0.012, y = 1.002, label = "Ideal (0, 1)", size = 3, color = "grey40", fontface = "bold", hjust = 0) +
  geom_point(size = 1.8, alpha = 0.5) +
  geom_segment(data = arrows, inherit.aes = FALSE,
               aes(x = x0, y = y0, xend = far, yend = sens, color = method),
               arrow = arrow(length = unit(0.22, "cm"), type = "closed"), linewidth = 0.9) +
  geom_point(data = c_mean, aes(x = far, y = sens, color = method), shape = 18, size = 5, inherit.aes = FALSE) +
  geom_text(data = arrows, inherit.aes = FALSE,
            aes(x = far, y = sens, label = sprintf("\u0394Sens %+.3f", dsens), color = method),
            nudge_y = 0.014, size = 2.8, fontface = "bold", hjust = 0.5) +
  geom_text(data = arrows, inherit.aes = FALSE,
            aes(x = far, y = sens, label = sprintf("\u0394FAR %+.3f", dfar), color = method),
            nudge_y = -0.014, size = 2.8, fontface = "bold", hjust = 0.5) +
  scale_color_manual(values = meth_col) +
  coord_cartesian(xlim = c(0, 0.45), ylim = c(0.86, 1.02)) +
  labs(x = "False Alarm Rate (0->1)", y = "Event Sensitivity (0->1)") + theme_pub +
  theme(legend.position = c(0.78, 0.18), legend.justification = c(0.5, 0.5))

# ============ d：时间序列（图例图内左上，2 周刻度，分组统计入 caption） ============
d01 <- d %>% filter(A == 0, B == 1)
d01$ts <- as.POSIXct(d01$timestamp); d01$ev <- factor(d01$y_true, levels = c(0,1), labels = c("Normal","Event"))
tau <- unique(d01$threshold)
r <- rle(d01$in_extreme_task); ends <- cumsum(r$lengths); starts <- ends - r$lengths + 1
seg <- data.frame(t0 = d01$ts[starts], t1 = d01$ts[ends], ext = r$values); seg <- seg[seg$ext == 1, ]
p_d <- ggplot(d01, aes(x = ts, y = distance)) +
  geom_rect(data = seg, inherit.aes = FALSE, aes(xmin = t0, xmax = t1, ymin = -Inf, ymax = Inf),
            fill = "#F0F0F0", alpha = 0.9) +
  # annotate("text", x = min(d01$ts) + as.numeric(difftime(max(d01$ts), min(d01$ts), units="secs"))/2,
  #          y = Inf, label = "Shaded: extreme-task periods",
  #          hjust = 0.5, vjust = 1.6, size = 3.2, fontface = "bold", color = "grey35") +
  geom_hline(yintercept = tau, linetype = "dashed", color = "black", linewidth = 0.8) +
  annotate("text", x = max(d01$ts), y = tau,
           label = sprintf("Threshold = %.3f (95th pct, calib. ref.)", tau),
           hjust = 1, vjust = -0.7, size = 3.2, fontface = "bold") +
  geom_point(aes(color = ev), size = 0.9, alpha = 0.7) +
  scale_color_manual(values = c("Normal" = "#8FA8C8", "Event" = "#D83C3C")) +
  scale_x_datetime(date_breaks = "2 weeks", date_labels = "%b %d") +
  labs(x = "Time (2005)", y = "Distance Score") +
  theme_pub +
  theme(legend.position = c(0.02, 0.99), legend.justification = c(0, 1),
        axis.text.x = element_text(face = "bold", size = 11),
        plot.caption = element_text(face = "bold", size = 9, hjust = 0, color = "grey25"))

# ============ 组合：a|b|c 上排，d 全宽下排 ============
fig <- (p_a | p_b | p_c) / p_d + plot_layout(heights = c(1, 0.9))
fig
# ggsave(file.path(REPO, "main_results/figures/fig45_cosine.png"), fig,
#        width = 14.5, height = 9.8, dpi = 300)