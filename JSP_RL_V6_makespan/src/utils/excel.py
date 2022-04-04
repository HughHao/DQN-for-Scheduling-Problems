#生成excel表格，导出数据
#输出每次的Q表，状态，选择的动作，观察是否按照Q表选择了Q值最小值
import xlsxwriter

def getExcel(RL,Qexcel,episode,start_row,num_action,observation,selectedActions):
    RL.q_table.to_excel(Qexcel, startrow=start_row, sheet_name='Sheet' + str(episode))
    worksheet = Qexcel.sheets['Sheet' + str(episode)]
    worksheet.write(start_row, 0, num_action)
    worksheet.write(start_row, len(RL.q_table.columns) + 1, 'current state')
    worksheet.write(start_row, len(RL.q_table.columns) + 2, 'selected action')
    worksheet.write(start_row, len(RL.q_table.columns) + 3, 'makespan')
    worksheet.write(start_row + 1, len(RL.q_table.columns) + 1, str(observation))
    worksheet.write(start_row + 1, len(RL.q_table.columns) + 2, selectedActions)
