#include <stdio.h>
#include "_List_H.h"


int IsEmpty( List L )
{
	return L->Next == NULL; 
} 

int IsLast( Position P, List L ) 
{
	return P->Next == NULL;
}

Position Find( ElementType X, List L ) 
{
	Position P;
	
	P = L->Next;
	while ( P!= NULL && P->Element != X ) 
		P = P->Next;
	return P;
}

void Delete( ElementType X, List L)
{
	Position P, TempCell;

	P = FindPrevious(X, L);
	if ( !IsLast(P, L))
	{
		TempCell = P->Next;
		P->Next = TempCell->Next;
		free(TempCell);
	}
}

Position FindPrevious(ElementType X, List L)
{
	Position P; 

	P = L;
	while( P->Next != NULL && P->Next->Element != X)
		P = P->Next;
	return P;
}

void Insert(ElementType X, List L, Position P)
{
	Position TmpCell;
	TmpCell = malloc( sizeof( struct Node) );
	if ( TmpCell == NULL)
		FatalError( "Out of space!!!" );
	TmpCell->Element = X;
	TmpCell->Next = P->Next;
	P->Next = TmpCell;
}

void DeleteList( List L)
{
	Position P, Tmp;

	P = L->Next;
	L->Next = NULL;
	while( P != NULL)
	{
		Tmp = P->Next;
		free(P);
		P = Tmp;
	}
}
