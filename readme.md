# FluentType

FluentType 是一个为 TypeScript 设计的强类型流式接口构建器，可让您创建富有表现力且类型安全的 API。

## 特点

- 💪 **强类型**：利用 TypeScript 的类型系统创建类型安全的流式接口
- 🔗 **可链式调用**：构建直观的、可链式的方法调用
- 🛠️ **灵活多变**：支持各种模式，如构建器、命令等
- 📝 **自文档化**：清晰而富有表现力的代码，无需额外文档

## 安装

```bash
npm install fluent_type
```

## 使用方法

```typescript
import { fluent } from 'fluent_type';

// 示例用法
const queryBuilder = fluent()
    .method('select', (fields: string[]) => ({ fields }))
    .method('from', (table: string) => ({ table }))
    .method('where', (condition: string) => ({ condition }))
    .method('orderBy', (field: string, direction: 'asc' | 'desc' = 'asc') => ({ field, direction }))
    .build();

const query = queryBuilder
    .select(['name', 'email'])
    .from('users')
    .where('age > 18')
    .orderBy('name', 'asc');
```

## 文档

查看完整文档以获取更多示例和高级用法。

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 许可证

[MIT](LICENSE)