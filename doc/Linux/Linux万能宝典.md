# 一、Linux学习路线

[linux B站视频](https://www.bilibili.com/video/BV1pE411C7ho/?spm_id_from=333.337.search-card.all.click&vd_source=1d204308936e108c95b2ecb8fcdbd781)

[Linux常用命令](https://segmentfault.com/a/1190000021950993)

[linux下载工具wget 与 curl 详解](https://segmentfault.com/a/1190000022301195)

[linux学习路线](https://zhuanlan.zhihu.com/p/420311740)

Linux命令大全
https://www.linuxcool.com/
https://www.runoob.com/linux/linux-command-manual.html
https://segmentfault.com/a/1190000021950993

[Linux菜鸟教程](https://www.runoob.com/linux/linux-file-attr-permission.html)

# 二、Linux安装及熟悉

[VMware虚拟机安装Ubuntu-2022最新版详细图文安装教程(VMware虚拟机安装+Ubuntu下载+VMware虚拟机配置运行](https://blog.csdn.net/qq_51646682/article/details/124787486)

[CentOS7的下载安装配置教程](https://blog.csdn.net/m0_51545690/article/details/123238360)


# 三、Linux常用命令

[Linux常用命令参考](https://segmentfault.com/a/1190000021950993)

## 1 命令的基本格式

1.1 命令的提示符

[root@localhost ~]#

* []：这是提示符的分隔符号，没有特殊含义。
* root：显示的是当前的登录用户，
  目前使用的是root用户登录。
* @：分隔符号，没有特殊含义。
* localhost：当前系统的简写主机名（完整主机名是
  localhost.localdomain）。
* ~：代表用户当前所在的目录，此例中用户当前所在的目录是home目录。
* #：命令提示符。超级用户是#，普通用户是$

1.2 命令的基本格式

[root@localhost ~]# 命令 [选项] [参数]

* 选项：是用于调整命令的功能的
* 参数：是命令的操作对象

## 2 目录操作命令

2.1 ls命令

ls 是最常见的目录操作命令，主要作用是显示目录下的内容

* 命令名称：ls
* 英文原意：list
* 所在路径：/bin/ls
* 执行权限：所有用户
* 功能描述：显示目录下的内容

命令格式：ls [选项] [目录名]

* 选项
  * -a:
    显示所有文件
  * --color=when:
    支持颜色输出，when的值默认是always(总显示颜色，文件跟文件夹颜色有区别)，也可以是never(从不显示颜色)和auto(自动)
  * -d:
    显示目录信息，而不是目录下的文件
  * -h:
    人性化显示，按照我们习惯的单位显示文件大小
  * -i:
    显示文件的节点号
  * -l: 长格式显示

示例：
```shell
[root@localhost~]$ ls -l
#权限 引用计数 所有者 所属组 大小 文件修改时间 文件名
-rw-------. 1 root root  1446 12月 19 16:15 anaconda-ks.cfg

“-l” 选项用于显示文件的详细信息，那么“-l”选项显示的这 7
列分别是什么含义？

第一列：权限。
第二列：引用计数。文件的引用计数代表该文件的硬链接个数，而目录的引用计数代表该目录有多少个一级子目录。
第三列：所有者，也就是这个文件属于哪个用户。默认所有者是文件的建立用户
第四列：所属组。默认所属组是文件建立用户的有效组，一般情况下就是建立用户的所在组。
第五列：大小。默认单位是字节。
第六列：文件修改时间。文件状态修改时间或文件数据修改时间都会更改这个时间，注意这个时间不是文件的创建时间。
第七列：文件名。
```
2.2 cd命令

cd是切换所在目录的命令，基本信息如下：

* 命令名称：cd
* 英文原意：change directory
* 所在路径：shell内置命令
* 执行权限：所有用户
* 功能描述：切换所在目录

命令格式：cd [-L|-P] [dir]

* 选项：
  * -L:（默认值）如果要切换到的目标目录是一个符号连接，那么切换到符号连接的目录。
  * -P:
    如果要切换到的目标目录是一个符号连接，那么切换到它指向的物理位置目录。
* 参数:
* 可以是绝对路径(以根目录为参照物)或相对路径(以当前目录为参照物)

简化用法:
```bash
cd ~     #当前用户的home目录
cd -     #上一次所在目录
cd .     #当前目录
cd ..    #上级目录
```
2.3 pwd命令

pwd命令是查询所在目录的命令，基本信息如下：

* 命令名称：pwd
* 英文原意：print name of current/working directory
* 所在路径：/bin/pwd
* 执行权限：所有用户
* 功能描述：查询当前所在目录

2.4 mkdir命令

* 命令名称：mkdir
* 英文原意：make directories
* 所在路径：/bin/mkdir
* 执行权限：所有用户
* 功能描述：创建空目录

命令格式：mkdir [选项] 目录名

* 选项：
  * -p: 递归建立所需目录
  * -m 权限：建立目录的同时设置目录的权限

2.5 rmdir命令

rmdir命令删除空目录，基本信息如下：

* 命令名称：rmdir
* 英文原意：remove empty directories
* 所在路径：/bin/rmdir
* 执行权限：所有用户
* 功能描述：删除空目录

命令格式：rmdir [选项] 目录名

* 选项：
  * -p: 递归删除目录

rmdir命令的作用十分有限，只能删除空目录，一旦目录中有内容就会报错。所以一般不论删除的是文件还是目录，都会使用rm命令

## 3 文件操作命令

3.1 touch命令

touch命令创建空文件或修改文件时间，基本信息如下：

* 命令名称：touch
* 英文原意：change file timestamps
* 所在路径：/bin/touch
* 执行权限：所有用户
* 功能描述：创建文件或改文件时间戳

命令格式：touch [选项] 文件名

* 选项：
  * -a：或--time=atime或--time=access或--time=use
    更改存取时间为当前时间(access time)
  * -m：或--time=mtime或--time=modify
    更该变动时间为当前时间(modify time)
  * -t 日期时间:
    使用指定的日期时间(格式：[[CC]YY]MMDDhhmm[.ss])，而非现在的时间
  * -r 参考文件或目录:
    把指定文件或目录的日期时间统统设成参考文件或目录的日期时间

3.2 cat命令

cat命令用来查看文件内容，基本信息如下：

* 命令名称：cat
* 英文原意：concatenate files and print on the standard output
* 所在路径：/bin/cat
* 执行权限：所有用户
* 功能描述：合并文件并打印输出到标准输出

命令格式：cat [选项] 文件名

* 选项：
  * -E: 列出每行结尾的回车符$
  * -n: 显示行号
  * -T: 把Tab键用^I显示出来
  * -v: 列出特殊字符
  * -A:
    相当于-vET选项的整合，用于列出所有隐藏符号

3.3 more 命令

more是分屏显示文件的命令，基本信息如下：

* 命令名称：more
* 英文原意：file perusal filter for crt viewin
* 所在路径：/bin/more
* 执行权限：所有用户
* 功能描述：分屏显示文件内容

命令格式：more [选项] 文件名

* 选项：
  * -d：显示“[press space to
    continue,'q' to quit.]”和“[Press 'h' for instructions]”；
  * -c：不进行滚屏操作。每次刷新这个屏幕
  * -s：将多个空行压缩成一行显示
  * -u：禁止下划线
  * -数字：指定每屏显示的行数
  * +数字：从指定数字的行开始显示

more
命令比较简单，一般不用什么选项，命令会打开一个交互界面，可以识别一些交互命令。常用的交互命令如下：

* 空格键：向下翻页
* B键: 向上翻页
* /字符串：搜索指定的字符串
* q: 退出

3.4 less 命令

less命令和more命令类似，只是more命令是分屏显示，而less是分行显示命名，less命令允许用户向前(PageUp键)或向后(PageDown键)浏览文件，基本信息如下：

* 命令名称：less
* 英文原意：opposite
  of more
* 所在路径：/usr/bin/more
* 执行权限：所有用户
* 功能描述：分行显示文件内容

命令格式：less [选项] 文件名

* -e：文件内容显示完毕后，自动退出
* -f：强制显示文件
* -g：不加亮显示搜索到的所有关键词，仅显示当前显示的关键字，以提高显示速度
* -l：搜索时忽略大小写的差异
* -N：每一行行首显示行号
* -s：将连续多个空行压缩成一行显示
* -S：在单行显示较长的内容，而不换行显示
* -x数字：将TAB字符显示为指定个数的空格字符

3.5 head 命令

head 命令是用来显示文件开头内容的命令，基本信息如下：

* 命令名称：head
* 英文原意：output
  the first part files
* 所在路径：/usr/bin/head
* 执行权限：所有用户
* 功能描述：显示文件开头的内容

命令格式：head [选项] 文件名

* -n 行数：从文件开头开始，显示指定行数
* -v：显示文件名

3.6 tail 命令

tail 命令是用来显示文件结尾内容的命令

* 命令名称：tail
* 英文原意：output
  the last part files
* 所在路径：/usr/bin/tail
* 执行权限：所有用户
* 功能描述：显示文件结尾的内容

命令格式：tail [选项] 文件名

* -n 行数：从文件结尾开始，显示指定行数
* -v：显示文件名
* -f: 监听文件新增内容

3.7 ln 命令

ln命令用来为文件创建链接，连接类型分为硬连接和符号连接(软链接)两种，基本信息如下：

* 命令名称：ln
* 英文原意：make links between file
* 所在路径：/bin/tail
* 执行权限：所有用户
* 功能描述：在文件之间建立链接

命令格式：ln [选项] 源文件 [目标文件]

* 选项：
  * -s:
    建立软链接文件。如果不加'-s'选项，则建立硬链接文件
  * -f:
    强行删除已存在的链接文件。如果链接文件已存在，则删除目标文件后再建立链接文件

  * 源文件：指定链接的源文件。如果使用-s选项创建软链接，则“源文件”可以是文件或者目录，创建硬链接时，则“源文件”参数只能是文件。（源文件最好用绝对路径名，这样可以在任何工作目录下进行符号链接，而当源文件用相对路径时，如果当前的工作路径与要创建的符号链接文件所在路径不同，就不能进行链接）
  * 目标文件：指定源文件的目标链接文件，省略则在当前目录下新建与源文件名称相同的链接文件

硬链接和源文件实际上是同一个文件，不会创建新的文件(类似于引用)；
而软链接会创建一个新文件来保存源文件的路径，从而间接读取或修改源文件内容

**硬链接与软链接的特征**

硬链接特征：
1.源文件和硬链接文件拥有相同的Indoe和Block
2.修改任意一个文件，另一个都改变
3.删除任意一个文件，另一个都能使用
4.硬链接建立或删除，原文件连接数相应加一或减一
5.硬链接不能链接目录
6.硬链接不能跨分区
7.硬链接标记不清，很难确认硬链接文件位置，不建议使用

软链接特征：
1.软链接和源文件拥有不同的Inode和Block
2.两个文件修改任意一个，另一个都改变
3.删除软链接，源文件不受影响；删除源文件，软链接不能使用
4.软链接建立或删除，原文件链接数不变
5.软链接可以链接目录
6.软链接可以跨分区
7.软链接特征明显，建议使用软链接
8.软链接没有实际数据，只是保存源文件的Inode，不论源文件多大，软链接大小不变
9.软链接的权限是最大权限lrwxrwxrwx.，但是由于没有实际数据，最终访问时需要参考源文件权限

## 4 文件和目录都能操作的命令

4.1 rm 命令

rm是最强大的删除命令，不仅可以删除文件，也可以删除目录，基本信息如下：

* 命令名称：rm
* 英文原意：remove files or directories
* 所在路径：/bin/rm
* 执行权限：所有用户
* 功能描述：删除文件或目录

命令格式：rm [选项] 文件或目录

* 选项：
  * -f: 强制删除
  * -i: 交互删除，在删除之前会询问用户
  * -r: 递归删除，可以删除目录

4.2 cp 命令

cp命令用于复制文件或目录，基本信息如下：

* 命令名称：cp
* 英文原意：copy files and directories
* 所在路径：/bin/cp
* 执行权限：所有用户
* 功能描述：复制文件或目录

命令格式：cp [选项] 源文件 目标文件

* 选项：
  * -d: 如果文件为软链接(对硬链接无效)，则复制出的目标文件也为软链接
  * -i: 询问，如果目标文件已经存在，则会询问是否覆盖
  * -p: 复制后目标文件保留源文件的属性(包括所有者、所有组、权限和时间)
  * -r: 递归复制，用于复制目录
  * -a: 相当于-dpr选项的集合

4.3 mv 命令

mv命令用来剪贴文件或目录，基本信息如下：

* 命令名称：mv
* 英文原意：move(rename) files
* 所在路径：/bin/mv
* 执行权限：所有用户
* 功能描述：移动文件或目录

命令格式：cp [选项] 源文件 目标文件

* 选项：
  * -f:
    强制覆盖，如果目标文件已经存在，则不询问直接强制覆盖
  * -i:
    交互模式，如果目标文件已经存在，则询问用户是否覆盖（默认选项）
  * -v：显示详细信息

4.4 stat命令

stat命令是查看文件详细的命令，基本信息如下：

* 命令名称：stat
* 英文原意：display file or file system status
* 所在路径：/usr/bin/stat
* 执行权限：所有用户
* 功能描述：显示文件或文件系统的详细信息

命令格式：stat [选项] 文件名

* 选项：
  * -f：显示文件系统状态而非文件状态
  * -t：以简洁方式输出信息

## 5 基本权限管理

5.1 权限介绍

使用ls命令时，长格式显示的第一列就是文件的权限，例如：

[root@localhost~]# ls -l install.log
-rw-r--r--. 1 root root 28425 11月 30 18:50 install.log

第一列的权限位(-rw-r--r--.)如果不计最后的"."
(这个点代表受SELinux安全上下文保护，这里暂时忽略不做介绍)，则共10位，这10位权限位的含义如下图所示：

![1693905149274](image/Linux万能宝典/1693905149274.png)

* 第1位：代表文件类型。Linux不像Windows使用扩展名表示文件类型，而是使用权限位的第一位表示文件类型。虽然Linux文件的种类不像Windows中那么多，但是分类也不少，详细情况可以使用“info ls”命令查看。这里列出一些常见的文件类型：
  * -: 普通文件
  * d:目录文件。Linux中一切皆文件，所以目录也是文件的一种
  * l: 软链接文件
  * b:块设备文件。这是一种特殊设备文件，存储设备都是这种文件，如分区文件/dev/sda1就是这种文件
  * c:字符设备文件。这也是特殊设备文件，输入设备一般都是这种文件，如鼠标、键盘等
  * p: 管道符文件。这是一种非常少见的特殊设备文件。
  * s:套接字文件。这也是一种特殊设备文件，一些服务支持socket访问就会产生这样的文件
* 第2~4位：代表文件所有者的权限
* r: 代表read，是读取权限
* w： 代表write，是写权限
* x: 代表execute，是执行权限
* 第5~7位：代表文件所属组的权限，同样拥有“rwx”权限
* 第8~10位：代表文件其他人的权限，同样拥有“rwx”权限

权限含义的解释：

读、写、执行权限对文件和目录的作用是不同的。

* 权限对文件的作用
  * 读(r)：对文件有读权限，代表可以读取文件中的数据。如果把权限对应到命令上，那么一旦对文件有读权限，就可以对文件执行cat、more、less、head、tail等文件查看命令
  * 写(w)：对文件有写权限，代表可以修改文件中的数据。如果把权限对应到命令上，那么一旦对文件有写权限，就可以对文件执行vim、echo等修改文件数据的命令。注意：对文件有写权限，是不能删除文件本身的，只能修改文件中的数据，如果想要删除文件，则需要对文件的上级目录拥有写权限。
  * 执行(x)：对文件有执行权限，代表文件可以运行。在Linux中，只要文件有执行权限，这个文件就是执行文件了，只是这个文件到底能不能正确执行，不仅需要看执行权限，还要看文件的代码是不是正确的语言代码。对文件来说，执行权限是最高权限
* 权限对目录的作用
* 读(r)：对目录有读权限，代表可以查看目录下的内容，也就是可以查看目录下有哪些文件和子目录。如果包权限对应到命令上，那么一旦对目录拥有了读权限，就可以在目录下执行ls命令查看目录下的内容了
* 写(w)：对目录有写权限，代表可以修改目录下的数据，也就是可以在目录中新建、删除、复制、剪贴子文件或子目录。如果把权限对应到命令上，那么一旦对目录拥有了写权限，就可以在目录下执行touch、rm、cp、mv等命令。对目录来说，写权限是最高权限
* 执行(x)：目录是不能运行的，那么对目录拥有执行权限，代表可以进入目录。如果把权限对应到命令上，那么一旦对目录拥有了执行权限，就可以对目录执行cd命令进入目录

5.2 chmod 命令

chmod用来修改文件的权限，基本信息如下：

* 命令名称：chmod
* 英文原意：change file mode bits
* 所在路径：/bin/chmod
* 执行权限：所有用户
* 功能描述：修改文件的权限模式

命令格式：chmod [选项] 权限模式 文件或目录

* 选项：
  * -R:
    递归设置权限，也就是给予目录中的所有文件设定权限
  * --reference=RFILE：使用参考文件或参考目录RFILE的权限来设置目标文件或目录的权限。

chmod命令的权限模式分为符号组合和八进制数组合

符号组合的格式是[ugoa][[+-=][permission]]，也就是[用户身份][[赋予方式][权限]]的格式。

* 用户身份

  * u：代表所有者(user)
  * g：代表所属组(group)
  * o：代表其他人(other)
  * a：代表全部身份(all)
* 赋予方式
* +：加入权限
* -：减去权限
* =：设置权限
* 权限
* r: 读取权限(read)
* w: 写权限(write)
* x: 执行权限(execute)

八进制数组合的格式是[0-7][0-7][0-7]三位数字组成(每一位数字都是权限之和)，第一位是所有者权限，第二位是所属组权限，第三位其他人权限

* r读取权限对应的数字是4
* w写权限对应的数字是2
* x执行权限对应的数字是1
* 例如读写权限rw八进制数表示 6

示例：
```
# 添加组用户的写权限
chmodg +w ./test.log

# 删除其他用户的所有权限。

chmodo= ./test.log

# 使得所有用户都没有写权限。

chmoda -w ./test.log

# 当前用户具有所有权限，组用户有读写权限，其他用户只有读权限。  chmodu=rwx,

g=rw, o=r ./test.log (等价的八进制数表示:chmod754
./test.log )

将目录以及目录下的文件都设置为所有用户拥有读写权限。注意，使用'-R'选项一定要保留当前用户的执行和读取权限，否则会报错！  chmod-R
a=rw ./testdir/

# 根据其他文件的权限设置文件权限。

chmod--reference=./1.log ./test.log
```
5.3 chown 命令

chown 命令用来修改文件和目录的所有者和所属组，基本信息如下：

* 命令名称：chown
* 英文原意：change
  file owner and group
* 所在路径：/bin/chown
* 执行权限：所有用户
* 功能描述：修改文件和目录的所有者和所属组

命令格式：chown [选项] 所有者[:所属组] 文件或目录

* 选项：
  * -R：递归设置权限，也就是给予子目录的所有文件设置权限
* 当省略 “:所属组”，仅改变文件所有者

注意：普通用户不能修改文件的所有者，哪怕自己是这个文件的所有者也不行。普通用户可以修改所有者是自己的文件权限。

5.4 umask 命令

umask命令用来显示或设置创建文件或目录的权限掩码。

我们需要先了解一下新建文件和目录的默认最大权限，对于文件来讲，新建文件的默认最大权限是666，没有执行权限，只是因为执行权限对文件来讲比较危险，不能再新建文件的时候默认赋予，而必须通过用户手工赋予；对于目录来讲，新建目录的默认最大权限是777，这是因为对目录而言，执行权限仅仅代表进入目录，所以即使新建目录时直接默认赋予也没有什么危险。

按照官方的标准算法，umask默认权限需要使用二进制进行逻辑与和逻辑非联合运算才可以得到正确的新建文件和目录的默认权限，这种方法既不好计算也不好理解，不推荐。我们这里按照权限字母来讲解umask权限的计算方。我们就按照默认的umask值是0022(等效于022)分别来计算一下新建文件和目录的默认权限，

· 文件的默认权限最大只能是666，而umask的值是022，则 rw-rw-rw- 减去 ----w--w-等于rw-r--r--，所以新建文件的默认权限是rw-r--r--
· 目录的默认权限最大是777，而umask的值是022，则 rwxrwxrwx 减去 ----w--w-等于rwxr-xr-x，所以新建目录的默认权限是rwxr-xr-x
· 同理，如果umask的值是033，新建文件的默认权限为 rw-rw-rw- 减去 ----wx-wx等于rw-r--r--

命令格式：umask [选项] [模式]

· 选项：
  · -p：输出的权限掩码可直接作为指令来执行
  · -S：以符号组合的方式输出权限掩码，不使用该选项时以八进制数的形式输出

示例：

以八进制数的形式输出权限掩码[root@localhosttmp]# umask0022# 
以八进制数的形式输出权限掩码，并作为指令来执行 [root@localhosttmp]# umask-pumask 0022# 以符号组合的方式输出权限掩码。[root@localhosttmp]# umask -Su=rwx,g=rx,o=rx

#上条命令以符号组合的方式输出权限掩码，输出的结果u=rwx,g=rx,o=rx转化为八进制数等于0755，#用八进制数来设置同样的权限，umask需要额外的执行减法"0777 - 0755"即0022[root@localhosttmp]# umask
0022# 为组用户添加写权限[root@localhosttmp]# umask g+w# 删除其他用户的写、执行权限[root@localhosttmp]# umask
o-wx# 赋值全部用户所有权限，等价于umask u=rwx,g=rwx,o=rwx[root@localhosttmp]# umask a=rwx#清除其他用户的读、写、执行权限[root@localhosttmp]# umask
o=

## 6 帮助命令

6.1 man 命令

man命令是最常见的帮助命令，也是Linux最主要的帮助命令，基本信息如下：

* 命令名称：man
* 英文原意：format and display the on-line manual pages
* 所在路径：/usr/bin/chown
* 执行权限：所有用户
* 功能描述：显示连机帮助手册

命令格式：man [选项] [章节] 命令

* 选项：
  * -f: 查看命令有哪些章节的帮助和简短描述信息，等价于whatis指令
  * -k:
    查看和命令相关的所有帮助

man命令交互快捷键：

* 上箭头：向上移动一行
* 下箭头：向下移动一行
* PgUP：向上翻一页
* PgDn：向下翻一页
* g：移动到第一页
* G：移动到最后一页
* q：退出
* /字符串：从当前向下搜索字符串
* ?字符串：从当前向上搜索字符串
* n：当搜索字符串时，可以用n键找到下一个字符串
* N：当搜索字符串时，使用N键反向查询字符串。也就是说，如果使用“/字符串”方式搜索，则N键表示向上搜索字符串；如果使用“?字符串”方式搜索，则N键表示向下搜索字符串

man手册章节：

* 1： 用户在shell环境可操作的命令或执行文件
* 2： 系统内核可调用的函数与工具等
* 3：
  一些常用的函数(function)与函数库(library)，大部分为C的函数库(libc)
* 4： 设备文件说明，通常在/dev下的文件
* 5： 配置文件或某些文件格式
* 6： 游戏帮助(个人版的Linux中是有游戏的)
* 7： 惯例与协议等，如Linux文件系统，网络协议，ASCII code等说明
* 8： 系统管理员可用的管理命令
* 9： 跟kernel有关的文件

man手册的格式：

* NAME: 命令名称及功能简要说明
* SYNOPSIS：用法说明，包括可用的选项

  * []：可选内容
  * <>：必选内容
  * a|b：二选一
  * {}：分组
  * ...：同意内容可出现多次
* DESCRIPTION：命令功能的详细说明，可能包括每一个选项的意义
* OPTIONS：说明每一项的意义
* EXAMPLES：使用示例
* FILES：此命令相关的配置文件
* AUTHOR：作者
* COPYRIGHT：版本信息
* REPORTTING BUGS：bug信息
* SEE ALSO：参考其他帮助

示例：

我们输入 man ls，它会在最左上角显示“LS（1）”，在这里，“LS”表示手册名称，而“（1）”表示该手册位于第一节章，同样，我们输 man ifconfig 它会在最左上角显示“IFCONFIG（8）”。也可以这样输入命令：“man
[章节号] 手册名称”。

man是按照手册的章节号的顺序进行搜索的，比如：man sleep 只会显示sleep命令的手册，如果想查看库函数sleep，就要输入：man 3 sleep

6.2 info 命令

info 命令的帮助信息是一套完整的资料，每个单独命令的man帮助信息只是这套完整资料的某一个区段(节点)，基本信息如下：

* 命令名称：info
* 英文原意：info
* 所在路径：/usr/bin/info
* 执行权限：所有用户
* 功能描述：显示一套完整的帮助信息资料

命令格式：info [选项] 参数

* 选项：
  * -d：添加包含info格式帮助文档的目录
  * -f：指定要读取的info格式的帮助文档
  * -n：指定首先访问的info帮助文件的节点
  * -o：输出被选择的节点内容到指定文件
* 参数：指定需要获得帮助的主题，可以是指令、函数以及配置文件

info命令交互快捷键

* 上箭头：向上移动一行
* 下箭头：向下移动一行
* PgUP：向上翻一页
* PgDn：向下翻一页
* Tab：在有“*”符号的节点间切换
* 回车：进入有“*” 符号的子页面，查看详细帮助信息
* u：进入上一层信息(回车是进入下一层信息)
* q：退出info帮助信息
* n：进入下一小节信息
* p：进入上一下节信息
* ?：查看帮助信息

6.3 help 命令

help 命令只能获取shell内置命令的帮助，基本信息如下：

* 命令名称：help
* 英文原意：help
* 所在路径：shell
  内置命令
* 执行权限：所有用户
* 功能描述：显示shell内置命令的帮助。可以使用shell内置命令type来区分内置命令与外部命令，对于外部命令的帮助信息只能使用man或者info命令查看

命令格式：help [选项] 内置命令

* 选项：
  * -d：显示内建命令的简要描述。
  * -m：按照man手册的格式输出内置命令的帮助信息。
  * -s：仅输出内建命令的命令格式。

示例：
```
# 以man手册的格式查看内置命令type的帮助信息

[root@localhost ~]# help-mtypeNAME

    type-
Displayinformation aboutcommand
type.

SYNOPSIS

    type[-afptP]
name [name ...]

 (省略。。。)

# 查看ls、help命令是否是内置命令

[root@localhost ~]# typelslsis aliased to `ls--color=auto'

[root@localhost ~]# typehelphelpis a shellbuiltin
```

6.4 --help 选项

绝大多数命令都可以使用--help选项来查看帮助，者也是一种获取帮助的方法。例如 ls --help，这种方法非常简单，输出的帮助信息基本上是man命令的信息简要版。

## 7 搜索命令

7.1 whereis 命令

whereis 是搜索命令的命令，也就是说whereis不能搜索普通文件，而只能搜索系统命令，基本信息如下：

* 命令名称：whereis
* 英文原意：locate
  the binary,source,and manual page files for a command
* 所在路径：/usr/bin/whereis
* 执行权限：所有用户
* 功能描述：查找二进制命令、源文件和帮助文档的路径

命令格式：whereis [选项] 参数

* 选项：
  * -b：只查找二进制文件
  * -B 目录：只在设置的目录下查找二进制文件
  * -m：只查找说明文件
  * -M 目录：只在设置的目录下查找说明文件
  * -s：只查找原始代码文件
  * -S 目录：只在设置的目录下查找原始代码文件
  * -f：不显示文件名前的路径名称

7.2 which 命令

which 也是搜索系统命令的命令，和whereis的区别在于，whereis命令可以在查找二进制命令的同时，查找帮助文档的位置，而which命令在查找到二进制命令的同时，如果这个命令有别名，则还可以查到别名命令。基本信息如下：

* 命令名称：which
* 英文原意：shows
  the full path of (shell) commands
* 所在路径：/usr/bin/which
* 执行权限：所有用户
* 功能描述：列出二进制命令路径和别名。which只会在环境变量$PATH设置的目录里查找符合条件的命令

命令格式：which [选项] 参数

7.3 find 命令

find命令用来在指定目录下查找文件，基本信息如下：

* 命令名称：find
* 英文原意：search
  for files in a directory hierarchy
* 所在路径：/bin/find
* 执行权限：所有用户
* 功能描述：在指定目录中搜索文件

命令格式：find [搜索路径] [选项]

* 搜索路径：省略则默认为当目录，相当于
  "find ."
* 选项：
  * -name 范本样式：按照文件名称搜索，支持通配符模糊查询
  * -iname 范本样式：此参数的效果和指定“-name”参数类似，但忽略字符大小写的差别
  * -inum inode编号：查找符合指定的inode编号的文件或目录
  * -path 范本样式：查找路径包含范本样式的文件或目录
  * -regex 范本样式：正则表达式搜索
  * -iregex 范本样式：同"-regex"，忽略大小写
  * -size [+|-]文件大小[cwbkMG] ：查找符合指定的文件大小的文件

    * "+"
      的意思是搜索比指定大小还要大的文件，"-" 的意思是搜索比指定大小还要小的文件
    * "cwbkMG"是单位，c——字节，w——字(2字节)，b——块(512字节)，k——千字节，M——兆字节，G——吉字节。如果不写单位默认是b
  * -atime [+|-]天数：按照文件最后一次访问时间搜索，单位每天
  * "+"、"-"的含义，例如"5"表示恰好5天前的那一天，"+5"超过5天前的时间，"-5"5天内的时间。(以下按时间搜索选项中"+"、"-"含义相同)
  * -mtime [+|-]天数：按照文件数据最后一次修改时间搜索，单位每天
  * -ctime [+|-]天数：按照文件元数据(如权限等)最后一次修改时间搜索，单位每天
  * -amin [+|-]分钟数：按照文件最后一次访问时间搜索，单位每分钟
  * -mmin [+|-]分钟数：按照文件数据最后一次修改时间搜索，单位每分钟
  * -cmin [+|-]分钟数：按照文件元数据(如权限等)最后一次修改时间搜索，单位每分钟
  * -perm [+|-]权限数值：查找符合指定的权限数值的文件或目录。例如，权限数值为"766"表示权限恰好等于766，"-766"表示文件权限必须全部包含766，"+766"表示文件权限包含766任意一个权限
  * -uid 用户ID：查找所有者是指定用户ID的文件
  * -user 用户名：查找所有者是指定用户名的文件
  * -gid 组ID：查找所有组是指定组ID的文件
  * -group 组名：查找所有组是指定组名的文件
  * -nouser：查找没有所有者的文件
  * 按照所有者和所有组搜索时，"-nouser"选项比较常用，主要用于查找垃圾文件。没有所有者的文件比较少见，那就是外来文件，比如光盘和U盘的文件是由Windows复制的，在Linux中查看就是没有所有者的文件，再比如手工源码包安装的文件也可能没有所有者
  * -type 文件类型：只寻找符合指定的文件类型的文件
  * f——普通文件，l——符号连接，d——目录，c——字符设备，b——块设备，s——套接字，p——Fifo
  * -empty：查找文件大小为0的文件
  * -maxdepth 目录层级数：设置搜索的最大目录层级
  * -mindepth 目录层级：设置搜索的最小目录层级
  * -exec 执行指令：把find命令查找结果交由"-exec"调用的命令来处理
  * 格式：find [搜索路径] [选项] -exec
    命令 {} \;, 其中"{}"代表find命令的查询结果
  * -ok 执行指令：此参数的效果和指定“-exec”类似，但在执行指令之前会先询问用户是否执行
  * -prune：不寻找字符串作为寻找文件或目录的范本样式
  * -a：and 逻辑与
  * -o: or 逻辑或
  * -not：not 逻辑非

示例：

#==================根据文件名或者正则表达式进行匹配=====================#列出当前目录及子目录下所有文件和文件夹[root@localhost
~]# find.

#在 `/home`目录下查找以.txt结尾的文件名[root@localhost ~]#
find/home -name "*.txt"#同上，但忽略大小写[root@localhost ~]# find/home -iname "*.txt"#当前目录及子目录下查找所有以.txt和.pdf结尾的文件[root@localhost ~]# find. \( -name "*.txt"-o
-name "*.pdf"\)

或

[root@localhost ~]# find. -name "*.txt"-o -name "*.pdf"#查找路径包含local的文件或者目录[root@localhost ~]# find/usr/ -path "*local*"#基于正则表达式匹配文件路径[root@localhost ~]# find. -regex ".*\(\.txt\|\.pdf\)$"#=====================借助 `-exec`选项与其他命令结合使用==================#找出当前目录下所有root的文件，并把所有权更改为用户tom[root@localhost
~]# find.-type f -user root -exec chown tom
{} \;

#找出自己家目录下所有的.txt文件并删除，删除前先询问[root@localhost
~]# find$HOME/.
-name "*.txt"-ok rm {} \;

#查找当前目录下所有.txt文件并把他们拼接起来写入到all.txt文件中[root@localhost
~]# find. -type f -name "*.txt"-exec cat {} \;> all.txt

#将30天前的.log文件移动到old目录中[root@localhost ~]# find. -type f -mtime +30 -name "*.log"-exec cp {} old \;

#找出当前目录下所有.txt文件并以“File:文件名”的形式打印出来[root@localhost
~]# find. -type f -name "*.txt"-exec printf "[File: %s\n](File://%20%25s/n)"{}
\;

#========================逻辑运算符==========================#查找文件大小超过2k并且是普通文件类型的文件[root@localhost
~]# find. -size +2k -a -type f

#找出/home下不是以.txt结尾的文件[root@localhost tmp]# find. -not -name "*.txt"或

[root@localhost ~]# find/home ! -name "*.txt"#======================搜索但跳出指定的目录===================#查找当前目录或者子目录下所有.txt文件，但是跳过子目录sk[root@localhost
~]# find. -path "./sk"-prune
-o -name "*.txt"

7.4 locate 命令

locate命令其实是find
-name的另一种写法，但是要比后者快得多，原因在于它不搜索具体目录(find 是去硬盘找)，而是搜索一个数据库/var/lib/mlocate/mlocate.db，这个数据库中含有本地所有文件信息。Linux系统自动创建这个数据库，并且每天自动更新一次，所以使用locate命令查不到最新变动过的文件，为了避免这种情况，可以在使用locate之前，先使用updatedb命令手动更新数据库。locate命令基本信息如下：

* 命令名称：locate
* 英文原意：find
  files by name
* 所在路径：/usr/bin/locate
* 执行权限：所有用户
* 功能描述：按照文件名搜索文件

命令格式：locate [选项] 文件名

* 选项：
  * -d:
    指定资料库的路径。默认是/var/lib/mlocate/mlocate.db
  * -n：至多显示n个输出

数据库配置文件(/etc/updatedb.conf)内容说明：

[root@localhost ~]# cat /etc/updatedb.conf#开启搜索限制，也就是让这个文件生效PRUNE_BIND_MOUNTS= "yes"#在locate执行搜索时，禁止搜索这些文件类型PRUNEFS=
"9p afs anon_inodefs auto autofs bdev
binfmt_misc cgroup cifs coda configfs cpuset debugfs devpts ecryptfs exofs fuse
fusectl gfs gfs2 gpfs hugetlbfs inotifyfs iso9660 jffs2 lustre mqueue ncpfs nfs
nfs4 nfsd pipefs proc ramfs rootfs rpc_pipefs securityfs selinuxfs sfs sockfs
sysfs tmpfs ubifs udf usbfs"#在locate执行搜索时，禁止搜索这些扩展名的文件PRUNENAMES= ".git
.hg .svn"##在locate执行搜索时，禁止搜索这些系统目录PRUNEPATHS= "/afs
/media /net /sfs /tmp /udev /var/cache/ccache /var/spool/cups /var/spool/squid
/var/tmp"

locate优缺点：

优点：按照数据库搜索，搜索速度快，消耗资源小

缺点：只能按照文件名来搜索文件，而不能执行更复杂的搜索，比如按照权限、大小、修改时间等

7.5 grep 命令

grep命令的作用是在文件中提取和匹配符合条件的字符串行，是一种强大的文本搜索工具，它能使用正则表达式搜索文本，并把匹配的行打印出来。基本信息如下：

* 命令名称：grep
* 英文原意：global
  search regular expression(RE) and print out the line
* 所在路径：/usr/bin/grep
* 执行权限：所有用户
* 功能描述：全面搜索正则表达式并把行打印出来

命令格式：grep [选项] 搜索内容 文件名

* 选项：
  * -i：忽略大小写
  * -n：输出行号
  * -v：反向查找
  * -d 动作:
    当指定要查找的是目录而非文件时，必须使用这项参数，否则grep指令将回报信息并停止动作。动作包含：read、recurse、skip
  * -R或-r: 此参数的效果和指定“-d
    recurse”参数相同，递归查找目录下的所有文件内容
  * --color=auto：搜素出的关键字用颜色高亮显示

find也是搜索命令，那么find与grep命令有什么区别呢？

find：find命令用于在系统中搜索符合条件的文件名，如果需要模糊查询，则使用通配符进行匹配，通配符是完全匹配。(find命令也可以通过"-regex"选项，把匹配规则转为正则表达式规则)

grep：grep命令用于在文件中搜索符合条件的字符串，如果需要模糊查询，则使用正则表达式进行匹配，正则表达式是包含匹配。

通配符与正则表达式的区别

* 通配符：一般用于匹配文件名，完全匹配
  * ?：匹配一个任意字符
  * *：匹配0个或多个任意字符，也就是可以匹配任何内容
  * []：匹配中括号里任意一个字符。例如，[abc]代表一定匹配一个字符，或是a，或是b，或是c
  * [-]：匹配中括号里任意一个字符，"-"代表一个范围。例如，[a-z]代表匹配一个小写字母
  * [^]：逻辑非，表示匹配不是中括号里的一个字符。例如，[^0-9]代表匹配一个不是数字的字符
* 正则表达式：一般用于匹配字符串
* ?：匹配前一个字符重复0次或1次
* *：匹配前一个字符重复0次或多次
* []：匹配中括号里任意一个字符。例如，[abc]代表一定匹配一个字符，或是a，或是b，或是c
* [-]：匹配中括号里任意一个字符，"-"代表一个范围。例如，[a-z]代表匹配一个小写字母
* [^]：逻辑非，表示匹配不是中括号里的一个字符。例如，[^0-9]代表匹配一个不是数字的字符
* ^：匹配行首
* $：匹配行尾

示例：

#在文件中搜索一个单词，命令会返回一个包含“match_pattern”的文本行：[root@localhost~]# grep
match_pattern file_name或

[root@localhost~]#
grep "match_pattern"file_name#在多个文件中查找：[root@localhost~]# grep "match_pattern"file_1 file_2 file_3#标记匹配颜色 --color=auto 选项：[root@localhost~]# grep "match_pattern"file_name
--color=auto#在当前目录中对文本进行递归搜索[root@localhost~]# grep -r "match_pattern".#正则匹配输出以数字开头的所有行[root@localhost~]# grep "^[0-9].*"file_name

7.6 | 管道符

命令格式：命令1 | 命令2

"|"管道符的作用是把命令1的正确输出作为命令2的操作对象

示例1：

我们经常使用 "ll"
命令查看文件的长格式，不过在有些目录中文件很多，不如/etc/目录使用 "ll"
命令显示的内容就非常多，只能看到最后的内容而不能看到前面输出的内容，这时我们马上想到 "more"
命令可以分屏显示文件内容，一种笨方法是：

#用输出重定向，把"ll"命令的输出保存到/root/testfile

[root@localhost ~]# ll -a /etc/> /root/testfile

#然后用more分屏显示

[root@localhost ~]# more /root/testfile

这样操作实在是不方便，这时可以利用管道符，命令如下：

#把"ll"命令的输出作为"more"命令的操作对象[root@localhost~]# ll -a /etc/ | more

示例2：

#在"ll"命令输出内容中搜索yum的文件名[root@localhost~]# ll -a /etc/ | grep yum

示例3：

#统计具体的网络连接数量（"grep"命令筛选，"wc"命令统计）[root@localhost~]# netstat -an | grep -i "ESTABLISHED"| wc -l

7.7 alias 命令

alias命令用来设置指令的别名，我们可以使用该命令可以将一些较长的命令进行简化。

alias基本使用方法：

* 打印已经设置的命令别名
  * alias 或 alias -p
* 给命令设置别名
* 格式：alias 新的命令='实际命令'。必须使用单引号''实际命令引起来，防止特殊字符导致错误
* 例如：alias l='ls -lsh'，现在只用输入 "l" 就可以列出目录了，相当于输入"ls -lsh"；alias ser='service network restart'，现在输入"ser"就可以重启网络服务了

直接在shell里设定的命令别名，在终端关闭或者系统重新启动后都会失效，如何才能永久有效呢？

使用编辑器打开~/.bashrc，在文件中加入别名设置，如：alias
rm='rm -i'，保存后执行source
~/.bashrc，这样就可以永久保存命令的别名了。因为修改的是当前用户目录下的~/.bashrc文件，所以这样的方式只对当前用户有用。如果要对所有用户都有效，修改/etc/bashrc文件就可以了。

## 8 压缩和解压缩命令

在Linux中可以识别的常见压缩格式有十几种，比如".zip"、".gz"、".bz2"、".tar"、".tar.gz"、".tar.bz2"等。减少文件大小有两个明显的好处，一是可以减少存储空间，二是通过网络传输文件时，可以减少传输的时间

8.1 ".zip" 格式

".zip"是Windows中最常见的压缩格式，Liunx也可以正确识别".zip"格式，这可以方便地和Windows系统通用压缩文件。

8.1.1 zip 命令

zip命令就是".zip"格式的压缩命令，基本信息如下：

* 命令名称：zip
* 英文原意：package
  and compress(archive) files
* 所在路径：/usr/bin/zip
* 执行权限：所有用户
* 功能描述：压缩文件和目录

命令格式：zip [选项] 压缩包名 源文件或目录

* 选项：
  * -r：压缩目录递归处理，将指定目录下的所有文件和子目录一并处理

8.1.2 unzip 命令

unzip命令就是".zip"格式的解压缩命令，基本信息如下：

* 命令名称：unzip
* 英文原意：list,
  test and extract compressed files in a zip archive
* 所在路径：/usr/bin/unzip
* 执行权限：所有用户
* 功能描述：列表、测试和提取压缩文件中的文件

命令格式：unzip [选项] 压缩包名

* 选项：
  * -d：指定解压的位置

8.2 ".gz" 格式

".gz"格式是Linux中最常见的压缩格式。

8.2.1 gzip 命令

gzip命令是".gz"格式的压缩和解压缩命令，既方便又好用。gzip不仅可以用来压缩大的、较少使用的文件以节省磁盘空间，还可以和tar命令一起构成Linux操作系统中比较流行的压缩文件格式。据统计，gzip命令对文本文件有60%～70%的压缩率。注意：gzip不会打包文件，压缩的过程源文件会对应变为".gz"格式的压缩文件(源文件被删除)，解压缩的过程将".gz"格式的压缩文件再恢复成对应的源文件。基本信息如下：

* 命令名称：gzip
* 英文原意：compress
  or expand files
* 所在路径：/bin/gzip
* 执行权限：所有用户
* 功能描述：压缩或解压缩
  ".gz" 格式的文件或目录

命令格式：gzip [选项] 源文件

* 选项：
  * -d：执行解压缩
  * -r：递归处理，将指定目录下的所有文件及子目录一并处理
  * -c：将压缩数据输出到标准输出中，可以保留源文件
    * 使用"-c"选项，压缩数据会直接输出到屏幕上，为了不让压缩数据输出到屏幕上而是重定向到压缩文件中，并且同时保留源文件，命令可以这样写：gzip -c abc > abc.gz
  * -l：列出压缩文件的相关信息

8.2.2 gunzip 命令

gunzip命令用来解压缩 ".gz" 格式的文件(即使用
"gzip" 命令压缩的文件)，作用相当于 "gzip -d 压缩文件"，因此不论是压缩或解压缩，都可通过
"gzip" 指令单独完成。基本下信息如下：

* 命令名称：gunzip
* 英文原意：expand
  files
* 所在路径：/bin/gunzip
* 执行权限：所有用户
* 功能描述：解压缩".gz"
  格式的文件或目录

命令格式：gunzip [选项] 压缩文件

* 选项：
  * -r：递归处理，将指定目录下的所有文件及子目录一并处理
  * -c：把解压后的文件数据输出到标准输出中，可以保留压缩文件
  * -l：列出压缩文件的相关信息

8.3 ".bz2" 格式

".bz2"
格式是Linux的另一种压缩格式，从理论上来讲，".bz2" 格式的算法更新进、压缩比更好；而 ".gz"
格式相对来讲压缩的时间更快

8.3.1 bzip2 命令

bzip2 命令是 ".bz2" 格式文件的压缩和解压缩命令。注意："bzip2"不能用来压缩目录。基本信息如下：

* 命令名称：bzip2
* 英文原意：a
  block-sorting file compressor
* 所在路径：/usr/bin/bzip2
* 执行权限：所有用户
* 功能描述：压缩或解压缩
  ".bz2" 格式的文件

命令格式：bzip2 [选项] 源文件

* 选项：
  * -d：执行解压缩
  * -k：压缩或解压缩后，会删除原始文件，若要保留原始文件，请使用此参数
  * -f：在压缩或解压缩时，若输出文件与现有文件同名，强制覆盖现有文件
  * -c：将压缩与解压缩的数据输出到标准输出中

8.3.2 bunzip2 命令

bunzip2命令用来解压缩 ".bz2" 格式的文件(即使用
"bzip2" 命令压缩的文件)，作用相当于 "bzip2 -d 压缩文件"，因此不论是压缩或解压缩，都可通过
"bzip2" 指令单独完成。基本信息如下：

* 命令名称：bunzip2
* 英文原意：a
  block-sorting file compressor
* 所在路径：/usr/bin/bunzip2
* 执行权限：所有用户
* 功能描述：解压缩
  ".bz2" 格式的文件

命令格式：bunzip2 [选项] 压缩文件

* 选项：
  * -k：bzip2在解压缩后，会删除原始压缩文件，若要保留原始压缩文件，请使用此参数
  * -f：解压缩时，若输出的文件与现有文件同名时，强制覆盖现有的文件
  * -c：将解压缩的数据输出到标准输出中

8.4
".tar"、".tar.gz"、".tar.bz2" 格式

tar命令可以把一大堆的文件和目录全部打包成一个文件，这对于备份文件或将几个文件组合成为一个文件以便于网络传输是非常有用的。注意：打包和压缩是两个不同的概念，打包是指将一大堆文件或目录变成一个总的文件；压缩则是将一个大的文件通过一些压缩算法变成一个小文件。为什么要区分这两个概念呢？这源于Linux中很多压缩程序(gzip、bzip2)只能针对一个文件进行压缩，这样当你想要压缩一大堆文件时，你得先将这一大堆文件先打成一个包（tar），然后再用压缩程序进行压缩。tar命令基本信息如下：

* 命令名称：tar
* 英文原意：tar
* 所在路径：/usr/bin/tar
* 执行权限：所有用户
* 功能描述：打包与解打包文件

命令格式：

打包：tar -c [选项] [-f 包文件名] 源文件或目录

解打包：tar -x [选项] -f 包文件名

* 选项：
  * -c：执行打包
  * -x：执行解打包
  * -z：支持压缩和解压缩
    ".tar.gz" 格式文件
  * -j：支持压缩和解压缩
    ".tar.bz2" 格式文件
  * -C 目录路径：指定解打包位置
  * -f 包文件名:
    指定打包文件名(.tar)或压缩包文件名(.tar.gz、.tar.bz2)。（执行打包时不写此选项，会默认把打包数据输出到屏幕）
  * -v: 显示打包或解打包过程
  * -t：测试，就是不解打包，只是查看包中有哪些文件

示例：

#=======================".tar"格式=========================#打包不会压缩[root@localhost~]# tar -cvf
anaconda-ks.cfg.tar anaconda-ks.cfg#解打包到当前目录[root@localhost~]# tar -xvf
anaconda-ks.cfg.tar#解打包到指定目录[root@localhost~]# tar -xvf anaconda-ks.cfg.tar -C
/testdir/#=====================".tar.gz"格式=====================#把/tmp/目录直接打包并压缩为".tar.gz"格式[root@localhost~]# tar
-zcvf tmp.tar.gz /tmp/#解压缩并解打包".tar.gz"格式文件[root@localhost~]# tar -zxvf tmp.tar.gz#=====================".tar.bz2"格式=====================#把/tmp/目录直接打包并压缩为".tar.bz2"格式[root@localhost~]# tar
-jcvf tmp.tar.gz /tmp/#解压缩并解打包".tar.bz2"格式文件[root@localhost~]# tar -jxvf tmp.tar.gz

## 9 关机和重启命令

9.1 sync 数据同步

sync命令用于强制被改变的内容立刻写入磁盘。在Linux/Unix系统中，在文件或数据处理过程中一般先放到内存缓冲区中，等到适当的时候再写入磁盘，以提高系统的运行效率。sync命令则可用来强制将内存缓冲区中的数据立即写入磁盘中。用户通常不需执行sync命令，系统会自动执行update或bdflush操作，将缓冲区的数据写入磁盘。只有在update或bdflush无法执行或用户需要非正常关机时，才需手动执行sync命令。基本信息如下：

* 命令名称：sync
* 英文原意：flush
  file system buffers
* 所在路径：/bin/sync
* 执行权限：所有用户
* 功能描述：刷新文件系统缓冲区

命令格式：sync [选项]

9.2 shutdown 命令

shutdown命令用来系统关机。shutdown指令可以关闭所有程序，并依用户的需要，进行重新开机或关机的动作。基本信息如下：

* 命令名称：shutdown
* 英文原意：bring
  the system down
* 所在路径：/sbin/shutdown
* 执行权限：超级用户
* 功能描述：关机和重启

命令格式：shutdown [选项] 时间 [警告信息]

* 选项：
  * -c：取消将要执行的shutdown命令
  * -h：系统关机
  * -r：系统重启
* 时间：now 立即执；hh:mm 指定确定时间点执行；+分钟数 延迟指定分钟后执行
* 警告信息：执行指令时，同时送出警告信息给登入用户

示例：

#立即关机[root@localhost~]# shutdown -h now#指定5分钟后关机，同时送出警告信息给登入用户：[root@localhost~]# shutdown
+5 "System will shutdown after 5
minutes"

9.3 reboot 命令

reboot命令用来重新系统，命令也是安全的，而且不需要过多的选项

* 命令名称：reboot
* 英文原意：reboot
* 所在路径：/sbin/reboot
* 执行权限：超级用户
* 功能描述：重启系统

命令格式：reboot [选项]

* 选项：
  * -d：重新开机时不把数据写入记录文件/var/tmp/wtmp。本参数具有“-n”参数效果；
  * -f：强制重新开机，不调用shutdown指令的功能；
  * -i：在重开机之前，先关闭所有网络界面；
  * -n：重开机之前不检查是否有未结束的程序；
  * -w：仅做测试，并不真正将系统重新开机，只会把重开机的数据写入/var/log目录下的wtmp记录文件。

9.4 halt 和 poweroff 命令

halt 和 poweroff
这两都是系统关机命令，直接执行即可。但是两个命令不会完整关闭和保存系统的服务，不建议使用。

9.5 init 命令

init 命令是修改Linux
运行级别的命令，是Linux下的进程初始化工具，init进程是所有Linux进程的父进程，它的进程号为1。也可以用于关机和重启，这个命令并不安全，不建议使用

命令格式：init [选项] 系统运行级别

* 选项：
  * -b：不执行相关脚本而直接进入单用户模式
  * -s：切换到单用户模式

示例：

#关机，也就是调用系统的0级别[root@localhost~]# init 0#重启，也就是调用系统的6级别[root@localhost~]# init 6

Linux有7个系统运行级别

| 运行级别 | 含义                                                      |
| -------- | --------------------------------------------------------- |
| 0        | 关机                                                      |
| 1        | 单用户模式，可以想象为Windows的安全模式，主要用于系统修复 |
| 2        | 不完全的命令模式，不含NFS服务                             |
| 3        | 完全的命令模式，就是标准字符界面                          |
| 4        | 系统保留，没有用到                                        |
| 5        | 图形模式                                                  |
| 6        | 重启动                                                    |

runlevel 命令可查看当前系统运行级别

## 10 常用网络命令

10.1 配置IP地址

IP地址是计算机在互联网中唯一的地址编码，每台计算机如果需要接入网络和其他计算机进行数据通信，就必须配置唯一的IP地址

IP地址的配置有两种方法：

* setup 工具
* 手工修改网卡配置文件
  * 第一步：编辑网卡文件，vi
    /etc/sysconfig/network-scripts/ifcfg-eth0，"ifcfg-eth0"是第一块网卡，第二块网卡则为"ifcfg-eth1"，以此类推。网卡文件内容配置项如下：
    * DEVICE=eth0
      #网卡设备名
    * BOOTPROTO=static
      #[none|static|bootp|dhcp]（引导时不使用协议|静态分配|BOOTP协议|DHCP协议）
    * HWADDR=00:15:5D:00:46:83
      #MAC地址
    * UUID=5753e2ed-add1-4d1c-8a69-21a89647b050# 唯一识别码
    * NM_CONTROLLED=yes
      #是否可以由Network Manager图形管理工具托管
    * ONBOOT=yes
      #[yes|no]，是否随网络服务启动，如果配置"no"，使用"ifconfig"命令时看不到该网卡
    * TYPE=Ethernet# 网络类型
    * IPADDR=192.168.1.10
      #IP地址
    * NETMASK=255.255.255.0
      #子网掩码
    * NETWORK=192.168.1.0
      #网络地址
    * BROADCAST=192.168.1.255
      #广播地址
    * GATEWAY=192.168.1.1
      #网关地址
    * DNS1=202.109.14.5
      #首选DNS服务地址
    * DNS2=219.141.136.10
      #备用DNS服务地址
    * USERCTL=no
      #[yes|no]（非root用户是否可以控制该设备）
  * 第二步：查看DNS服务配置文件，vim /etc/resolv.conf，里面的内容是系统自动生成的，一般不需要修改
  * 第三步：重启网络服务，service network restart 或 /etc/init.d/network
    restart

注意：使用虚拟机克隆时，UUID可能复制的是一样的，导致网络服务启动失败，需要重置UUID值：

第一步：编辑网卡文件，删除UUID和MAC地址

第二步: 删除MAC地址和UUID绑定文件，rm -rf /etc/udev/rules.d/70-persistent-net.rules

第三部：重启系统

10.2 ifconfig 命令

ifconfig 被用于配置和显示Linux内核中网络接口的网络参数。用ifconfig命令配置的网卡信息，在网卡重启后机器重启后，配置就不存在，要想将上述的配置信息永远的存的电脑里，那就要修改网卡的配置文件了。基本信息如下：

* 命令名称：ifconfig
* 英文原意：configure
  a network interface
* 所在路径：/sbin/ifconfig
* 执行权限：超级用户
* 功能描述：配置网络接口

命令格式：ifconfig [参数]

* 参数：
  * add 地址：设置网络设备IPv6的ip地址；
  * del 地址：删除网络设备IPv6的IP地址；
  * down：关闭指定的网络设备；
  * <hw<网络设备类型><硬件地址>：设置网络设备的类型与硬件地址；
  * io_addr I/O地址：设置网络设备的I/O地址；
  * irq IRQ地址：设置网络设备的IRQ；
  * media 网络媒介类型：设置网络设备的媒介类型；
  * mem_start 内存地址：设置网络设备在主内存所占用的起始地址；
  * metric 数目：指定在计算数据包的转送次数时，所要加上的数目；
  * mtu 字节：设置网络设备的MTU；
  * netmask 子网掩码：设置网络设备的子网掩码；
  * tunnel 地址：建立IPv4与IPv6之间的隧道通信地址；
  * up：启动指定的网络设备；
  * IP地址：指定网络设备的IP地址；
  * 网络设备：指定网络设备的名称。

ifconfig 命令最主要的作用就是查看IP地址的信息，直接输入ifconfig命令即可显示激活状态的网络设备信息:

[root@localhost
~]# ifconfig

eth0      Link encap:Ethernet  HWaddr 00:16:3E:00:1E:51inet addr:10.160.7.81Bcast:10.160.15.255Mask:255.255.240.0UP
BROADCAST RUNNING MULTICAST  MTU:1500Metric:1RX
packets:61430830errors:0dropped:0overruns:0frame:0TX
packets:88534errors:0dropped:0overruns:0carrier:0collisions:0txqueuelen:1000RX bytes:3607197869(3.3GiB)  TX
bytes:6115042(5.8MiB)

loLink encap:Local Loopback

    inet addr:127.0.0.1Mask:255.0.0.0UP
LOOPBACK RUNNING  MTU:16436Metric:1RX
packets:56103errors:0dropped:0overruns:0frame:0TX packets:56103errors:0dropped:0overruns:0carrier:0collisions:0txqueuelen:0RX
bytes:5079451(4.8MiB)  TX bytes:5079451(4.8MiB)

内容说明：

* eth0 表示第一块网卡，其中HWaddr表示网卡的物理地址，可以看到目前这个网卡的物理地址(MAC地址）是00:16:3E:00:1E:51
* inet addr 用来表示网卡的IP地址，此网卡的IP地址是10.160.7.81，广播地址Bcast:10.160.15.255，掩码地址Mask:255.255.240.0
* lo 是表示主机的回环地址，这个一般是用来测试一个网络程序，但又不想让局域网或外网的用户能够查看，只能在此台主机上运行和查看所用的网络接口。比如把
  httpd服务器的指定到回环地址，在浏览器输入127.0.0.1就能看到你所架WEB网站了，但只是本机能看得到，局域网的其它主机或用户无从知道
  * 第一行：连接类型：Ethernet（以太网）HWaddr（硬件mac地址）
  * 第二行：网卡的IP地址、子网、掩码
  * 第三行：UP（代表网卡开启状态）RUNNING（代表网卡的网线被接上）MULTICAST（支持组播）MTU:1500（最大传输单元）：1500字节
  * 第四、五行：接收、发送数据包情况统计
  * 第七行：接收、发送数据字节数统计信息

其他示例：

#启动和关闭网卡eth0。关闭网卡eth0，ssh登陆linux服务器操作要小心，关闭了就不能开启了，除非你有多网卡[root@localhost~]# ifconfig
eth0 up[root@localhost~]# ifconfig eth0 down#为网卡eth0配置IPv6地址[root@localhost~]# ifconfig
eth0 add 33ffe:3240:800:1005::2/64
#为网卡eth0删除IPv6地址[root@localhost~]# ifconfig eth0 del 33ffe:3240:800:1005::2/64
#用ifconfig修改MAC地址[root@localhost~]# ifconfig eth0 hw ether 00:AA:BB:CC:dd:EE#ifconfig配置IP地址[root@localhost~]# ifconfig
eth0 192.168.2.10[root@localhost~]# ifconfig eth0 192.168.2.10 netmask 255.255.255.0[root@localhost~]# ifconfig
eth0 192.168.2.10 netmask 255.255.255.0 broadcast 192.168.2.255#启用和关闭arp协议[root@localhost~]# ifconfig
eth0 arp   [root@localhost~]# ifconfig
eth0 -arp#设置最大传输单元，这里设置能通过的最大数据包大小为 1500 bytes[root@localhost~]# ifconfig
eth0 mtu 1500

10.3 ping 命令

ping 命令是常用的网络命令，主要通过ICMP协议进行网络探测，测试网络中主机的通信情况。基本信息如下：

* 命令名称：ping
* 英文原意：send
  ICMP ECHO_REQUEST to network hosts
* 所在路径：/bin/ping
* 执行权限：所有用户
* 功能描述：向网络主机发送ICMP请求

命令格式： ping
[选项] IP

* 选项：
  * -b：用于对整个网段进行探测。IP需要为广播地址
  * -c
    次数：设置完成要求回应的次数
  * -s
    字节数：设置数据包的大小

示例：

#探测整个网段中有多少主机是可以和本机通信的，而不是一个一个IP地址地进行探测[root@localhost
~]# ping -b -c 3
192.168.199.255

WARNING: pinging broadcast address

PING 192.168.199.255 (192.168.199.255) 56(84) bytes of data.

64 bytes from192.168.199.216: icmp_seq=1 ttl=64
time=77.7 ms

64 bytes from192.168.199.131: icmp_seq=2 ttl=64
time=102 ms

64 bytes from192.168.199.216: icmp_seq=3 ttl=64
time=19.5 ms

---

192.168.199.255 ping statistics ---

3 packets transmitted, 3 received, 0% packet loss, time 2023ms

rtt min/avg/max/mdev = 19.515/66.701/102.798/34.892 ms

10.4 netstat 命令

netstat是网络状态查看命令，既可以查看到本机开启的端口，也可以查看哪些客户端连接。在CentOS
7.x中 netstat 命令默认没有安装，如果需要使用，需要先安装 "net-snmp" 和 "net-tools"
软件包。基本信息如下：

* 命令名称：network
* 英文原意：print
  network connections, routing tables, interface statistics, masquerade
  connections, and mulicast memberships
* 所在路径：/bin/netstat
* 执行权限：所有用户
* 功能描述：打印网络连接、路由表、接口统计信息、伪装连接和多播成员身份

命令格式：netstat [选项]

* 选项：
  * -a：列出所有网络状态，包括Socket程序
  * -c 秒数：制定每隔几秒刷一次网络状态
  * -n：使用IP地址和端口号显示，不使用域名与服务名
  * -p：显示PID和程序
  * -t：显示使用TCP协议端口的连接情况
  * -u：显示使用UDP协议端口的连接情况
  * -l：仅显示监听状态的连接
  * -r：显示路由表

示例 1：查看本机开启的端口

这是本机最常用的方式，使用"-tuln"选项。因为使用了"-l"选项，所以只能看到监听状态的连接，而不能看到已经建立连接状态的连接，

[root@localhost~]# netstat
-tulnActive Internet connections (only servers)

Proto Recv-Q Send-Q Local Address
Foreign Address
State

tcp        000.0.0.0:33060.0.0.0:*LISTEN

tcp        000.0.0.0:112110.0.0.0:*LISTEN

tcp        000.0.0.0:220.0.0.0:*LISTEN

tcp        00:::11211:::*LISTEN

tcp        00:::80:::*LISTEN

tcp        00:::22:::*LISTEN

udp        000.0.0.0:112110.0.0.0:*udp
00:::11211:::*

这个命令输出内容较多，下面对每列进行说明：

* Proto：网络连接的协议，一般就是 TCP 协议或者 UDP 协议
* Recv-Q：接收队列。表示接收到的数据，已经在本地的缓冲中，但是还没有被进程取走
* Send-Q：发送队列。表示从本机发送，对方还没有收到的数据，依然在本地的缓冲中，一般是不具备ACK标志的数据包
* Local Address：本机的 IP 地址和端口号。
* Foreign Address：远程主机的 IP 地址和端口号。
* State：状态。常见的状态主要有以下几种：
  * LISTEN：监听状态，只有
    TCP 协议需要监听，而 UDP 协议不需要监听
  * ESTABLISHED：已经建立连接的状态。如果使用“-l”选项，则看不到已经建立连接的状态
  * SYN_SENT：SYN
    发起包，就是主动发起连接的数据包
  * SYN_RECV：接收到主动连接的数据包
  * FIN_WAIT1：正在中断的连接
  * FIN_WAIT2：已经中断的连接，但是正在等待对方主机进行确认
  * TIME_WAIT：连接已经中断，但是套接字依然在网络中等待结束
  * CLOSED：套接字没有被使用
  * 在这些状态中，我们最常用的就是
    LISTEN 和 ESTABLISHED 状态，一种代表正在监听，另一种代表已经建立连接

示例 2：查看本机有哪些程序开启的端口

如果使用“-p”选项，查询结果会多出一列"PID/Program
name"，则可以查看到是哪个程序占用了端口，并且可以知道这个程序的 PID

[root@localhost~]# netstat
-tulnpActive Internet connections (only servers)

Proto Recv-Q Send-Q Local Address
Foreign Address     State      PID/Program name

tcp        000.0.0.0:33060.0.0.0:*LISTEN
2359/mysqld

tcp        000.0.0.0:112110.0.0.0:*LISTEN
1563/memcached

tcp        000.0.0.0:220.0.0.0:*LISTEN
1490/sshd

tcp        00:::11211:::*LISTEN
1563/memcached

tcp        00:::80:::*LISTEN
21025/httpd

tcp        00:::22:::*LISTEN
1490/sshd

udp        000.0.0.0:112110.0.0.0:*1563/memcached

udp        00:::11211:::*1563/memcached

示例 3：查看所有连接

使用选项“-an”可以查看所有连接，包括监听状态的连接（LISTEN）、已经建立连接状态的

连接（ESTABLISHED）、Socket 程序连接等。因为连接较多，所以输出的内容有很多

[root@localhost
~]# netstat -anActive Internet connections
(only servers)

Proto Recv-Q Send-Q Local Address
Foreign Address
State

tcp        0  0 0.0.0.0:3306                0.0.0.0:*                   LISTEN

tcp        0  0 0.0.0.0:11211               0.0.0.0:*                   LISTEN

tcp        0  0 117.79.130.170:80           78.46.174.55:58815          SYN_RECV

tcp        0  0 0.0.0.0:22                  0.0.0.0:*                   LISTEN

tcp        0  0 117.79.130.170:22           124.205.129.99:10379        ESTABLISHED

tcp        0  0 117.79.130.170:22           124.205.129.99:11811
ESTABLISHED

...省略部分内容...

udp        0  0 0.0.0.0:11211               0.0.0.0:*

udp        0  0 :::11211                    :::*

Active UNIX domain sockets (servers and established)

Proto RefCnt Flags       Type       State         I-Node Path

unix  2 [
ACC ]     STREAM     LISTENING     12668 /var/run/mcelog-client

unix  2 [
ACC ]     STREAM     LISTENING     12193 @/var/run/hald/dbus-ZeYsMXZ7Uf

...省略部分内容...

从 "Active UNIX domain sockets" 开始，之后的内容就是
Socket 程序产生的连接，之前的内容都是网

络服务产生的连接。我们可以在“-an”选项的输出中看到各种网络连接状态，而之前的“-tuln”选项则只能看到监听状态的连接

10.5 write 命令

write
命令用于向指定登录用户终端上发送信息。通过write命令可传递信息给另一位登入系统的用户，当输入完毕后，键入 "回车" 表示发送，键入
"Ctrl+C" 表示信息结束。如果接收信息的用户不只登入本地主机一次，你可以指定接收信息的终端机编号。基本信息：

* 命令名称：write
* 英文原意：send
  a message to another user
* 所在路径：/usr/bin/write
* 执行权限：所有用户
* 功能描述：向其他用户发送消息

命令格式 ：write 用户名 [终端编号]

* 用户名：指定要接受信息的登录用户
* 终端编号：指定接收信息的用户的登录终端，如果省略，且用户在多个终端登录，会发送给其中一个终端

10.6 wall 命令

wall命令用于向系统当前所有打开的终端上输出信息，而 "write"
命令用于给指定用户发送消息。

命令格式：wall 消息

示例：

[root@localhost~]# wall "I will be in 5 minutes to restart, please save your
data"

## 11 系统痕迹命令

系统中有一些重要的痕迹日志文件，如 /var/log/wtmp、 /var/run/utmp、
/var/log/btmp、/var/log/lastlog 等日志文件，如果你用 vim
打开这些文件，你会发现这些文件是二进制乱码。这是由于这些日志中保存的是系统的重要登录痕迹，包括某个用户何时登录了系统，何时退出了系统，错误登录等重要的系统信息。这些信息要是可以通过
vim 打开，就能编辑，这样痕迹信息就不准确，所以这些重要的痕迹日志，只能通过对应的命令来进行查看。

11.1 w 命令

w 命令是显示系统中正在登录的用户信息的命令，这个命令查看的痕迹日志是
"/var/run/utmp"。基础信息如下：

* 命令名称：w
* 英文原意：show
  who is logged on and what they are doing
* 所在路径：/usr/bin/w
* 执行权限：所有用户
* 功能描述：显示登录用户和他们正在做什么

命令格式：w [选项] [用户名]

* 选项：
  * -h：不打印头信息；
  * -u：当显示当前进程和cpu时间时忽略用户名；
  * -s：使用短输出格式；
  * -f：显示用户从哪登录；
* 用户：仅显示指定用户

示例：

[root@localhost
~]# w18:31:08up 1day, 16min,  3users,  load average: 0.00,
0.00, 0.00USERTTYFROM              LOGIN@   IDLE
JCPU   PCPU WHAT

root     tty1     -                Sat18   49:230.07s0.07s-bash

root     pts/0192.168.199.11918:228:170.02s0.02s-bash

root     pts/1192.168.199.11918:220.00s0.03s0.00sw

说明：

* 第一行内容：
  * 18:31:08：系统当前时间
  * up 1 day,
    16 min：系统的运行时间
  * 3
    users：当前登录终端数量
  * load
    average: 0.00, 0.00,
    0.00：系统在之前1分钟、5分钟、15分钟的平均负载。如果CPU是单核的，则这个数据超过1就是高负载；如果CPU是四核的，则这个数值超过4就是高负载
* 第二行内容：
* USER：当前登录的用户
* TTY：登录的终端。tty1-6:
  本地字符终端(alt+F1-6 切换)，tty7: 本地图形终端(ctrl+F7切换，必须安装启动图形界面)，pts/0-255: 远程终端
* FROM：登录的IP地址，如果是本地终端，则是空
* LOGIN@：登录时间
* IDLE：用户空闲时间
* JCPU：所有的进程占用的CPU时间
* PCPU：当前进程占用的CPU时间
* WHAT：用户正在进行的操作

11.2 who 命令

who 命令和 w 命令类似，用于查看正在登录的用户，但显示的内容更加简单，也是查看
"/var/run/utmp" 日志。

命令格式：who [选项] [查询文件]

* 选项：
  * -H：显示各栏位的标题信息列
  * -q：只显示登入系统的帐号名称和总人数
  * -w：显示用户的信息状态栏
  * -u：显示闲置时间，若该用户在前一分钟之内有进行任何动作，将标示成"."号，如果该用户已超过24小时没有任何动作，则标示出"old"字符串
* 查询文件：指定要查询的文件，默认是/var/run/utmp

11.3 last 命令

last 命令查看系统所有登录过的用户信息，包括正在登录的用户和之前登录的用户，这个命令查看的是
"/var/log/wtmp" 痕迹日志文件

命令格式：last [选项] [用户|终端]

* 选项：
  * -a：把从何处登入系统的主机名称或ip地址，显示在最后一行
  * -d：将IP地址转换成主机名称
  * -f 记录文件：指定记录文件
  * -n 显示列数或-显示列数：设置列出名单的显示列数
  * -R：不显示登入系统的主机名称或IP地址
  * -x：显示系统关机，重新开机，以及执行等级的改变等信息
* 用户|终端：显示指定的用户或终端

11.4 lastlog 命令

lastlog 命令是查看系统中所有用户最好一次的登录时间的命令，这个命令查看的是
"/var/log/lastlog" 痕迹日志文件

命令格式：lastlog [选项]

* 选项：
  * -b 天数：显示指定天数前的登录信息
  * -t 天数：显示指定天数以来的登录信息
  * -u 用户名：显示指定用户的最近登录信息

11.5 lostb 命令

lastb 命令是查看错误登录的信息的，查看的是 "/var/log/btmp"
痕迹日志

命令格式：lostb [选项] [用户|终端]

* 选项：
  * -a：把从何处登入系统的主机名称或ip地址显示在最后一行
  * -d：将IP地址转换成主机名称
  * -f 记录文件：指定记录文件
  * -n 显示列数或-显示列数：设置列出名单的显示列数
  * -R：不显示登入系统的主机名称或IP地址
  * -x：显示系统关机，重新开机，以及执行等级的改变等信息
* 用户|终端：显示指定的用户或终端
