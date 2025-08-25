#!/bin/bash

# 强化学习模型可视化推理启动器

clear
echo "========================================"
echo "强化学习模型可视化推理启动器"
echo "========================================"
echo

# 函数：显示菜单
show_menu() {
    echo "请选择要运行的操作："
    echo "1. 快速测试 (推荐先运行)"
    echo "2. 可视化推理 (Walker2d环境)"
    echo "3. 可视化推理 (Hopper环境)"
    echo "4. 可视化推理 (Halfcheetah环境)"
    echo "5. 自定义参数运行"
    echo "6. 退出"
    echo
}

# 函数：快速测试
quick_test() {
    echo
    echo "运行快速测试..."
    python quick_test.py
    echo
    read -p "按Enter键继续..."
}

# 函数：Walker2d环境
walker2d() {
    echo
    echo "运行Walker2d环境可视化推理..."
    python visualize_inference.py --env walker2d --dataset medium --num_episodes 3
    echo
    read -p "按Enter键继续..."
}

# 函数：Hopper环境
hopper() {
    echo
    echo "运行Hopper环境可视化推理..."
    python visualize_inference.py --env hopper --dataset medium --num_episodes 3
    echo
    read -p "按Enter键继续..."
}

# 函数：Halfcheetah环境
halfcheetah() {
    echo
    echo "运行Halfcheetah环境可视化推理..."
    python visualize_inference.py --env halfcheetah --dataset medium --num_episodes 3
    echo
    read -p "按Enter键继续..."
}

# 函数：自定义参数
custom() {
    echo
    echo "请输入自定义参数："
    read -p "环境名称 (walker2d/hopper/halfcheetah/maze2d/antmaze): " env
    read -p "数据集类型 (medium/medium-replay/medium-expert/expert): " dataset
    read -p "Episode数量: " episodes
    
    echo
    echo "运行自定义参数可视化推理..."
    python visualize_inference.py --env "$env" --dataset "$dataset" --num_episodes "$episodes"
    echo
    read -p "按Enter键继续..."
}

# 主循环
while true; do
    clear
    show_menu
    read -p "请输入选择 (1-6): " choice
    
    case $choice in
        1)
            quick_test
            ;;
        2)
            walker2d
            ;;
        3)
            hopper
            ;;
        4)
            halfcheetah
            ;;
        5)
            custom
            ;;
        6)
            echo
            echo "感谢使用！"
            exit 0
            ;;
        *)
            echo
            echo "无效选择，请重新输入！"
            read -p "按Enter键继续..."
            ;;
    esac
done
