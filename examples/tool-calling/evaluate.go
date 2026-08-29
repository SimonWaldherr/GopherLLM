package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
)

// evaluate computes a +, -, *, /, parentheses arithmetic expression.
//
// It reuses go/parser to get correct operator precedence and parenthesis
// handling for free — go/parser parses arbitrary Go source, but evalNode's
// switch only ever descends into numeric literals, unary +/-, parenthesized
// expressions, and the four arithmetic binary operators; anything else
// (an identifier, a call, a string) is rejected before it is ever evaluated,
// not silently executed. No third-party expression-evaluator package is
// needed for what is otherwise a small hand-written recursive-descent parser.
func evaluate(expr string) (float64, error) {
	node, err := parser.ParseExpr(expr)
	if err != nil {
		return 0, fmt.Errorf("invalid expression %q: %w", expr, err)
	}
	return evalNode(node)
}

func evalNode(n ast.Expr) (float64, error) {
	switch v := n.(type) {
	case *ast.BasicLit:
		if v.Kind != token.INT && v.Kind != token.FLOAT {
			return 0, fmt.Errorf("unsupported literal %q", v.Value)
		}
		var f float64
		if _, err := fmt.Sscanf(v.Value, "%g", &f); err != nil {
			return 0, fmt.Errorf("invalid number %q: %w", v.Value, err)
		}
		return f, nil

	case *ast.ParenExpr:
		return evalNode(v.X)

	case *ast.UnaryExpr:
		x, err := evalNode(v.X)
		if err != nil {
			return 0, err
		}
		switch v.Op {
		case token.SUB:
			return -x, nil
		case token.ADD:
			return x, nil
		default:
			return 0, fmt.Errorf("unsupported unary operator %q", v.Op)
		}

	case *ast.BinaryExpr:
		x, err := evalNode(v.X)
		if err != nil {
			return 0, err
		}
		y, err := evalNode(v.Y)
		if err != nil {
			return 0, err
		}
		switch v.Op {
		case token.ADD:
			return x + y, nil
		case token.SUB:
			return x - y, nil
		case token.MUL:
			return x * y, nil
		case token.QUO:
			if y == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			return x / y, nil
		default:
			return 0, fmt.Errorf("unsupported operator %q", v.Op)
		}

	default:
		return 0, fmt.Errorf("unsupported expression syntax")
	}
}
