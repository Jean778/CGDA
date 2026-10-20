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
# cg_sup 相对 cal_sup 的宏平均 F1 增益哑铃图
# ============================================================
# 1. 安装并加载依赖
packages <- c("ggplot2", "patchwork")
missing_packages <- packages[
  !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}
library(ggplot2)
library(patchwork)

# 2. 可修改设置
input_file <- paste0(
  file.path(REPO, "main_results/"),
  "panel_correct_protocols_long.csv"
)
budget_use <- 0.01
output_dir <- file.path(dirname(input_file), "dumbbell_figures")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

gain_colour <- "#258C88"
loss_colour <- "#CA704C"
baseline_colour <- "#454F5B"

# 🟥 字号设置：所有字体最小 14pt，最大 16pt
# geom_text() 的 size 单位为 mm，theme 字号单位为 pt
pt_to_mm <- 25.4 / 72.27
font_value <- 10 * pt_to_mm       # 哑铃数值（14pt）
font_transfer <- 12 * pt_to_mm    # 迁移方向（14pt）
font_month <- 10 * pt_to_mm        # 月份（14pt）
font_gain <- 12 * pt_to_mm         # 增益（16pt）
font_column <- 12 * pt_to_mm       # Transfer 列标题（14pt）
font_gain_header <- 12 * pt_to_mm  # Gain 列标题（14pt）
font_axis_tick <- 12               # 横轴刻度（14pt）
font_axis_title <- 14              # 横轴标题（16pt）
font_subtitle <- 12                # 说明文字（14pt）
font_main_title <- 5              # 主标题（16pt）
font_panel_title <- 4.5             # 面板标题（14pt）

# 3. 读取与检查数据
dat <- read.csv(input_file, stringsAsFactors = FALSE)
required_columns <- c(
  "protocol", "source_batch", "target_batch",
  "direction", "budget", "seed", "cal_sup", "cg_sup"
)
if (!all(required_columns %in% names(dat))) {
  stop(
    "缺少列：",
    paste(setdiff(required_columns, names(dat)), collapse = ", ")
  )
}
key_columns <- c("protocol", "direction", "budget", "seed")
if (anyDuplicated(dat[key_columns])) {
  stop("存在重复的 protocol-direction-budget-seed 记录。")
}
dat <- dat[
  abs(dat$budget - budget_use) < 1e-8 &
    dat$protocol %in% c("Forward", "LOBO"),
]
if (nrow(dat) == 0) {
  stop("当前预算没有对应的数据。")
}
if (any(!is.finite(dat$cal_sup)) ||
    any(!is.finite(dat$cg_sup))) {
  stop("cal_sup 或 cg_sup 存在缺失值或非有限数值。")
}
if (any(dat$cal_sup < 0 | dat$cal_sup > 1) ||
    any(dat$cg_sup < 0 | dat$cg_sup > 1)) {
  stop("代码要求宏平均 F1 为 0–1，请检查数据量纲。")
}

# 4. 批次对应月份
batch_months <- c(
  "1,2",
  "3,4,8,9,10",
  "11,12,13",
  "14,15",
  "16",
  "17,18,19,20",
  "21",
  "22,23",
  "24,30",
  "36"
)

# 5. 对相同协议和方向进行种子配对汇总
groups <- split(
  dat,
  interaction(dat$protocol, dat$direction, drop = TRUE)
)
summary_dat <- do.call(
  rbind,
  lapply(groups, function(x) {
    
    stopifnot(
      length(unique(x$source_batch)) == 1,
      length(unique(x$target_batch)) == 1
    )
    
    paired_gain <- 100 * (x$cg_sup - x$cal_sup)
    
    data.frame(
      protocol = x$protocol[1],
      source_batch = x$source_batch[1],
      target_batch = x$target_batch[1],
      direction = x$direction[1],
      budget = budget_use,
      n_seeds = nrow(x),
      cal_mean = 100 * mean(x$cal_sup),
      cg_mean = 100 * mean(x$cg_sup),
      gain_mean = mean(paired_gain),
      gain_sd = sd(paired_gain)
    )
  })
)
rownames(summary_dat) <- NULL
summary_dat <- summary_dat[
  order(summary_dat$protocol, summary_dat$target_batch),
]
forward_rows <- summary_dat$protocol == "Forward"
fixed_rows <- summary_dat$protocol == "LOBO"
if (!any(forward_rows) || !any(fixed_rows)) {
  stop("需要同时包含 Forward 和 LOBO 两组数据。")
}
if (any(
  summary_dat$source_batch[forward_rows] !=
  summary_dat$target_batch[forward_rows] - 1
)) {
  stop("Forward 中存在非相邻批次迁移，请核对协议。")
}
if (any(summary_dat$source_batch[fixed_rows] != 1)) {
  stop("LOBO 组并非全部以 B1 为源域，请核对协议。")
}
if (any(!summary_dat$target_batch %in% 2:10)) {
  stop("目标批次超出 B2–B10。")
}

# 两个面板使用相同的性能坐标范围
axis_min <- max(
  0,
  floor(min(summary_dat$cal_mean, summary_dat$cg_mean) / 5) * 5 - 3
)
axis_max <- 100
axis_span <- axis_max - axis_min
axis_breaks <- pretty(c(axis_min, axis_max), n = 5)
axis_breaks <- axis_breaks[
  axis_breaks >= axis_min & axis_breaks <= axis_max
]

# 左侧文字列和右侧增益列的位置
label_x <- axis_min - 0.08 * axis_span
gain_x <- axis_max + 0.13 * axis_span
plot_limits <- c(
  axis_min - 0.25 * axis_span,
  axis_max + 0.20 * axis_span
)
title_left_x <- plot_limits[1] + 0.025 * axis_span

# 6. 单个面板绘图函数
make_panel <- function(z, panel_title) {
  
  z <- z[order(z$target_batch), ]
  
  # B2 最上方，B10 最下方
  z$y <- 11.8 - z$target_batch
  
  z$transfer_label <- sprintf(
    "B%d \u2192 B%d",
    z$source_batch,
    z$target_batch
  )
  
  z$month_label <- paste0(
    "M",
    batch_months[z$target_batch]
  )
  
  z$change <- ifelse(z$gain_mean >= 0, "Gain", "Loss")
  
  z$gain_label <- ifelse(
    abs(z$gain_mean) < 0.005,
    "\u22480.00",
    sprintf("%+.2f", z$gain_mean)
  )
  
  ggplot(z, aes(y = y)) +
    
    # 行参考线
    geom_hline(
      yintercept = 1:9,
      colour = "#EEEEEE",
      linewidth = 0.3
    ) +
    
    # 哑铃连接线
    geom_segment(
      aes(
        x = cal_mean,
        xend = cg_mean,
        yend = y,
        colour = change
      ),
      linewidth = 2.2,
      alpha = 0.65
    ) +
    
    # 基线：空心圆
    geom_point(
      aes(x = cal_mean),
      shape = 21,
      size = 2.7,
      fill = "white",
      colour = baseline_colour,
      stroke = 0.65
    ) +
    
    # 所提方法：实心圆，边框与填充同色
    geom_point(
      aes(
        x = cg_mean,
        fill = change,
        colour = change
      ),
      shape = 21,
      size = 2.7,
      stroke = 0.35
    ) +
    
    # 基线数值标在上方
    geom_text(
      aes(
        x = cal_mean,
        label = sprintf("%.1f", cal_mean)
      ),
      nudge_y = 0.5,
      size = font_value,
      colour = baseline_colour
    ) +
    
    # 所提方法数值标在下方
    geom_text(
      aes(
        x = cg_mean,
        label = sprintf("%.1f", cg_mean)
      ),
      nudge_y = -0.4,
      size = font_value,
      colour = baseline_colour
    ) +
    
    # 左侧：迁移方向
    geom_text(
      aes(x = label_x, label = transfer_label),
      hjust = 1,
      size = font_transfer,
      fontface = "bold"
    ) +
    
    # 左侧：目标批次月份
    geom_text(
      aes(x = label_x, label = month_label),
      hjust = 1,
      nudge_y = -0.50,
      size = font_month,
      colour = "#737D85"
    ) +
    
    # 右侧：配对增益
    geom_text(
      aes(
        x = gain_x,
        label = gain_label,
        colour = change
      ),
      size = font_gain,
      fontface = "bold"
    ) +
    
    # 面板标题
    annotate(
      "text",
      x = title_left_x,
      y = 12.05,
      label = panel_title,
      hjust = 0,
      size = font_panel_title,
      fontface = "bold",
      family = "sans"
    ) +
    
    # Transfer 列标题
    annotate(
      "text",
      x = title_left_x,
      y = 11,
      label = "Transfer / target months",
      hjust = 0,
      size = font_column,
      colour = baseline_colour
    ) +
    
    # Gain 列标题
    annotate(
      "text",
      x = gain_x,
      y = 11,
      label = "Gain (pp)",
      size = font_gain_header,
      fontface = "bold"
    ) +
    
    scale_colour_manual(
      values = c(Gain = gain_colour, Loss = loss_colour)
    ) +
    
    scale_fill_manual(
      values = c(Gain = gain_colour, Loss = loss_colour)
    ) +
    
    scale_x_continuous(
      limits = plot_limits,
      breaks = axis_breaks,
      expand = c(0, 0)
    ) +
    
    scale_y_continuous(
      limits = c(0.05, 12.55),
      breaks = NULL,
      expand = c(0, 0)
    ) +
    
    labs(
      x = "Macro-F1 (%)",
      y = NULL
    ) +
    
    theme_minimal(
      base_family = "sans",
      base_size = 12
    ) +
    
    theme(
      panel.grid = element_blank(),
      legend.position = "none",
      plot.margin = margin(8, 8, 8, 8),
      axis.text.x = element_text(size = font_axis_tick),
      axis.title.x = element_text(size = font_axis_title)
    )
}

# 7. 两个面板
p_forward <- make_panel(
  summary_dat[forward_rows, ],
  "Forward"
)
p_fixed <- make_panel(
  summary_dat[fixed_rows, ],
  "LOBO"
)

# 8. 顶部标题区域
p_header <- ggplot() +
  
  annotate(
    "text",
    x = title_left_x,
    y = 0.86,
    label = sprintf(
      "Target-domain gains at %g%% label budget",
      100 * budget_use
    ),
    hjust = 0,
    size = font_main_title,
    fontface = "bold",
    family = "sans"
  ) +
  
  scale_x_continuous(
    limits = plot_limits,
    expand = c(0, 0)
  ) +
  
  scale_y_continuous(
    limits = c(0, 1.08),
    expand = c(0, 0)
  ) +
  
  theme_void(base_family = "sans") +
  
  theme(
    plot.margin = margin(8, 8, 0, 8)
  )

# 三个绘图区共享列宽
figure <- (p_header / p_forward / p_fixed) +
  plot_layout(heights = c(0.5, 10.5, 10.5))

# 9. 在 RStudio 中显示
print(figure)






# 10. 保存图和汇总表
file_stem <- file.path(
  output_dir,
  sprintf("gain_budget_%02d", round(100 * budget_use))
)
ggsave(
  paste0(file_stem, ".png"),
  plot = figure,
  width = 7.2,
  height = 9.2,
  units = "in",
  dpi = 300,
  bg = "white"
)
if (capabilities("cairo")) {
  ggsave(
    paste0(file_stem, ".pdf"),
    plot = figure,
    width = 7.2,
    height = 9.2,
    units = "in",
    device = grDevices::cairo_pdf
  )
}
write.csv(
  summary_dat,
  paste0(file_stem, "_summary.csv"),
  row.names = FALSE
)
message("输出目录：", normalizePath(output_dir))
