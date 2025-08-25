@echo off
chcp 65001 >nul

echo ========================================
echo 强化学习模型可视化推理启动器
echo ========================================
echo.

echo 正在激活 conda 环境...
call conda activate decision-transformer-gym

if errorlevel 1 (
    echo 错误：无法激活 conda 环境 'decision-transformer-gym'
    echo 请检查环境名称是否正确
    pause
    exit /b 1
)

echo 环境激活成功！
echo.

:menu
echo 请选择要运行的操作：
echo 1. 快速测试 (推荐先运行)
echo 2. 可视化推理 (Walker2d环境)
echo 3. 可视化推理 (Hopper环境)
echo 4. 可视化推理 (HalfCheetah环境)
echo 5. 自定义参数运行
echo 6. 退出
echo.

set /p choice=请输入选择 (1-6): 

if "%choice%"=="1" goto quick_test
if "%choice%"=="2" goto walker2d
if "%choice%"=="3" goto hopper
if "%choice%"=="4" goto halfcheetah
if "%choice%"=="5" goto custom
if "%choice%"=="6" goto exit
goto invalid

:quick_test
echo.
echo 运行快速测试...
python quick_test.py
echo.
pause
goto menu

:walker2d
echo.
echo 运行Walker2d环境可视化推理...
python visualize_inference.py --env walker2d --dataset medium-replay --num_episodes 3 --save_gif True
echo.
pause
goto menu

:hopper
echo.
echo 运行Hopper环境可视化推理...
python visualize_inference.py --env hopper --dataset medium --num_episodes 3 --save_gif True
echo.
pause
goto menu

:halfcheetah
echo.
echo 运行HalfCheetah环境可视化推理...
python visualize_inference.py --env halfcheetah --dataset medium --num_episodes 3 --save_gif True
echo.
pause
goto menu

:custom
echo.
echo 请输入自定义参数：
set /p env=环境名称 (walker2d/hopper/halfcheetah/maze2d/antmaze): 
set /p dataset=数据集类型 (medium/medium-replay/medium-expert/expert): 
set /p episodes=Episode数量: 
set /p save_gif=是否保存GIF (True/False): 
echo.
echo 运行自定义参数可视化推理...
python visualize_inference.py --env %env% --dataset %dataset% --num_episodes %episodes% --save_gif %save_gif%
echo.
pause
goto menu

:invalid
echo.
echo 无效选择，请重新输入！
pause
goto menu

:exit
echo.
echo 正在退出 conda 环境...
call conda deactivate
echo 感谢使用！按任意键退出...
pause >nul
exit